// {{{ MIT License
//
// Copyright 2020 Roland Kaminski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//
// }}}

#include "clingcon.h"
#include "clingcon/propagator.hh"
#include "clingcon/dlpropagator.hh"
#include "clingcon/parsing.hh"

#include <clingo.hh>
#include <stdexcept>
#include <sstream>
#include <map>

#define CLINGCON_TRY try // NOLINT
#define CLINGCON_CATCH catch (...){ Clingo::Detail::handle_cxx_error(); return false; } return true // NOLINT

using Clingo::Detail::handle_error;

using namespace Clingcon;

namespace {

constexpr uint32_t MAX_THREADS = 64;
enum class Target { Heuristic, SignValue, RefineReasons, RefineIntroduce, PropagateChain, SplitAll };

} // namespace

struct clingcon_theory {
    Clingcon::Propagator propagator;
    ClingoDL::Stats dlstats;
    ClingoDL::Stats dlstep;
    ClingoDL::Stats dlaccu;
    ClingoDL::PropagatorConfig dlconf;
    ClingoDL::DifferenceLogicPropagator dlpropagator{dlstats, dlconf};
    Clingo::Detail::ParserList parsers;
    std::map<std::pair<Target, std::optional<uint32_t>>, val_t> deferred;
    bool shift_constraints{true};

    void on_dl_statistics(UserStatistics& step, UserStatistics &accu) {
        dlaccu.accu(dlstep);
        add_statistics(step, dlstep);
        add_statistics(accu, dlaccu);
        dlstep.reset();
    }

    static void add_statistics(UserStatistics& root, ClingoDL::Stats const &stats) {
        UserStatistics diff = root.add_subkey("DifferenceLogic", StatisticsType::Map);
        diff.add_subkey("Time init(s)", StatisticsType::Value).set_value(stats.time_init.count());
        diff.add_subkey("Mutexes", StatisticsType::Value).set_value(stats.mutexes);
        UserStatistics threads = diff.add_subkey("Thread", StatisticsType::Array);
        threads.ensure_size(stats.dl_stats.size(), StatisticsType::Map);
        auto it = threads.begin();
        for (ClingoDL::DLStats const& stat : stats.dl_stats) {
            auto thread = *it++;
            thread.add_subkey("Propagation(s)", StatisticsType::Value).set_value(stat.time_propagate.count());
            thread.add_subkey("Dijkstra(s)", StatisticsType::Value).set_value(stat.time_dijkstra.count());
            thread.add_subkey("Undo(s)", StatisticsType::Value).set_value(stat.time_undo.count());
            thread.add_subkey("True edges", StatisticsType::Value).set_value(stat.true_edges);
            thread.add_subkey("False edges", StatisticsType::Value).set_value(stat.false_edges);
            thread.add_subkey("Edges added", StatisticsType::Value).set_value(stat.edges_added);
            thread.add_subkey("Edges skipped", StatisticsType::Value).set_value(stat.edges_skipped);
            thread.add_subkey("Edges propagated", StatisticsType::Value).set_value(stat.edges_propagated);
            thread.add_subkey("Cost consistency", StatisticsType::Value).set_value(stat.propagate_cost_add);
            thread.add_subkey("Cost forward", StatisticsType::Value).set_value(stat.propagate_cost_from);
            thread.add_subkey("Cost backward", StatisticsType::Value).set_value(stat.propagate_cost_to);
        }
    }



};

namespace {

template<typename T>
bool init(clingo_propagate_init_t* c_init, void* data) {
    CLINGCON_TRY {
        Clingo::PropagateInit init{c_init};
        static_cast<T*>(data)->init(init);
    }
    CLINGCON_CATCH;
}

template<typename T>
bool propagate(clingo_propagate_control_t* c_ctl, const clingo_literal_t *changes, size_t size, void* data) {
    CLINGCON_TRY {
        Clingo::PropagateControl ctl{c_ctl};
        static_cast<T*>(data)->propagate(ctl, {changes, size});
    }
    CLINGCON_CATCH;
}

template<typename T>
void undo(clingo_propagate_control_t const *c_ctl, clingo_literal_t const *changes, size_t size, void* data) {
    Clingo::PropagateControl ctl(const_cast<clingo_propagate_control_t *>(c_ctl)); // NOLINT
    static_cast<T*>(data)->undo(ctl, {changes, size});
}

template<typename T>
bool check(clingo_propagate_control_t *c_ctl, void* data) {
    CLINGCON_TRY {
        Clingo::PropagateControl ctl{c_ctl};
        static_cast<T*>(data)->check(ctl);
    }
    CLINGCON_CATCH;
}

template<typename T>
bool decide(clingo_id_t thread_id, clingo_assignment_t const *c_ass, clingo_literal_t fallback, void* data, clingo_literal_t *result) {
    CLINGCON_TRY {
        Clingo::Assignment ass{c_ass};
        *result = static_cast<T*>(data)->decide(thread_id, ass, fallback);
    }
    CLINGCON_CATCH;
}

char const *flag_str(bool value) {
    return value ? "yes" : "no";
}

char const *heuristic_str(Clingcon::Heuristic heu) {
    switch(heu) {
        case Clingcon::Heuristic::None: {
            return "none";
        }
        case Clingcon::Heuristic::MaxChain: {
            return "max-chain";
            break;
        }
    };
    return "";
}

template <typename... Args>
[[nodiscard]] std::string format(Args &&... args) {
    std::ostringstream oss;
    (oss << ... << std::forward<Args>(args));
    return oss.str();
}

template<class T>
[[nodiscard]] T strtonum(char const *begin, char const *end) {
    if (!end) {
        end = begin + std::strlen(begin); // NOLINT
    }
    T ret = 0;
    bool sign = false;
    auto const *it = begin;
    if constexpr (std::is_signed_v<T>) {
        if (*it == '-') {
            sign = true;
            ++it; // NOLINT
        }
    }
    else {
        static_cast<void>(sign);
    }
    if (it == end) {
        throw std::invalid_argument("integer expected");
    }
    for (; it != end; ++it) { // NOLINT
        if ('0' <= *it && *it <= '9') {
            ret = safe_add<T>(safe_mul<T>(ret, 10), *it - '0'); // NOLINT
        }
        else {
            throw std::invalid_argument("integer expected");
        }
    }
    if constexpr (std::is_signed_v<T>) {
        return sign ? safe_inv<T>(ret) : ret;
    }
    else {
        return ret;
    }
}

template<class T, T min=std::numeric_limits<T>::min(), T max=std::numeric_limits<T>::max()>
[[nodiscard]] T parse_num(char const *begin, char const *end = nullptr) {
    auto res = strtonum<T>(begin, end);
    if (min <= res && res <= max) {
        return res;
    }
    throw std::invalid_argument("invalid argument");
}

template<class T, T min=std::numeric_limits<T>::min(), T max=std::numeric_limits<T>::max()>
[[nodiscard]] std::function<bool (const char *)> parser_num(T &dest) {
    return [&dest](char const *value) {
        dest = parse_num<T>(value);
        return true;
    };
}

void set_value(Target target, SolverConfig &config, val_t value) {
    switch (target) {
        case Target::SignValue: {
            config.sign_value = value;
            break;
        }
        case Target::Heuristic: {
            config.heuristic = static_cast<Clingcon::Heuristic>(value);
            break;
        }
        case Target::RefineReasons: {
            config.refine_reasons = value != 0;
            break;
        }
        case Target::RefineIntroduce: {
            config.refine_introduce = value != 0;
            break;
        }
        case Target::PropagateChain: {
            config.propagate_chain = value != 0;
            break;
        }
        case Target::SplitAll: {
            config.split_all = value != 0;
            break;
        }
    }
}

void set_value(Target target, Config &config, std::pair<val_t, std::optional<uint32_t>> const &value) {
    auto const &[val, thread] = value;
    if (thread.has_value()) {
        set_value(target, config.solver_config(*thread), val);
    }
    else {
        set_value(target, config.default_solver_config, val);
        for (auto &sconf : config.solver_configs) {
            set_value(target, sconf, val);
        }
    }
}

[[nodiscard]] bool parse_bool(char const *begin, char const *end = nullptr) {
    size_t len = end != nullptr ? end - begin : std::strlen(begin);
    if (std::strncmp(begin, "true", len) == 0 || std::strncmp(begin, "yes", len) == 0 || std::strncmp(begin, "1", len) == 0) {
        return true;
    }
    if (std::strncmp(begin, "false", len) == 0 || std::strncmp(begin, "no", len) == 0 || std::strncmp(begin, "0", len) == 0) {
        return false;
    }
    throw std::invalid_argument("invalid argument");
}


static char const *iequals_pre(char const *a, char const *b) {
    for (; *a && *b; ++a, ++b) {
        if (tolower(*a) != tolower(*b)) { return nullptr; }
    }
    return *b ? nullptr : a;
}
static bool iequals(char const *a, char const *b) {
    a = iequals_pre(a, b);
    return a && !*a;
}
static char const *parse_uint64_pre(const char *value, void *data) {
    auto &res = *static_cast<uint64_t*>(data);
    char const *it = value;
    res = 0;

    for (; *it; ++it) {
        if ('0' <= *it && *it <= '9') {
            auto tmp = res;
            res *= 10;
            res += *it - '0';
            if (res < tmp) { return nullptr; }
        }
        else { break; }
    }

    return value != it ? it : nullptr;
}
static bool parse_uint64(const char *value, void *data) {
    value = parse_uint64_pre(value, data);
    return value && !*value;
}

template <typename F, typename G>
bool set_config(char const *value, void *data, F f, G g) {
    try {
        auto &config = *static_cast<ClingoDL::PropagatorConfig*>(data);
        uint64_t id = 0;
        if (*value == '\0') {
            f(config);
            return true;
        }
        else if (*value == ',' && parse_uint64(value + 1, &id) && id < 64) {
            g(config.ensure(id));
            return true;
        }
    }
    catch (...) { }
    return false;
}

static bool parse_mutex(const char *value, void *data) {
    auto &pc = *static_cast<ClingoDL::PropagatorConfig*>(data);
    uint64_t x = 0;
    if (!(value = parse_uint64_pre(value, &x))) { return false; }
    pc.mutex_size = x;
    if (*value == '\0') {
        pc.mutex_cutoff = 10 * x;
        return true;
    }
    if (*value == ',') {
        if (!parse_uint64(value+1, &x)) { return false; }
        pc.mutex_cutoff = x;
    }
    return true;
}
static bool parse_mode(const char *value, void *data) {
    ClingoDL::PropagationMode mode = ClingoDL::PropagationMode::Check;
    char const *rem = nullptr;
    if ((rem = iequals_pre(value, "no"))) {
        mode = ClingoDL::PropagationMode::Check;
    }
    return rem && set_config(rem, data,
        [mode](ClingoDL::PropagatorConfig &config) { config.mode = mode; },
        [mode](ClingoDL::ThreadConfig &config) { config.mode = {true, mode}; });
}

[[nodiscard]] char const *find_str(char const *s, char c) {
    if (char const *t = std::strchr(s, c); t != nullptr) {
        return t;
    }
    return s + std::strlen(s); // NOLINT
}

[[nodiscard]] std::pair<val_t, std::optional<uint32_t>> parse_bool_thread(char const *value) {
    std::optional<uint32_t> thread = std::nullopt;
    char const *comma = find_str(value, ',');
    if (*comma != '\0') {
        thread = parse_num<uint32_t, 0, MAX_THREADS - 1>(comma + 1); // NOLINT
    }

    return {parse_bool(value, comma) ? 1 : 0, thread};
}

[[nodiscard]] std::pair<val_t, std::optional<uint32_t>> parse_sign_value(char const *value) {
    std::optional<uint32_t> thread = std::nullopt;
    char const *comma = find_str(value, ',');
    if (*comma != '\0') {
        thread = parse_num<uint32_t, 0, MAX_THREADS - 1>(comma + 1); // NOLINT
    }

    if (std::strncmp(value, "+", comma - value) == 0) {
        return {std::numeric_limits<val_t>::max(), thread};
    }
    if (std::strncmp(value, "-", comma - value) == 0) {
        return {std::numeric_limits<val_t>::min(), thread};
    }
    return {parse_num<val_t>(value, comma), thread};
}

[[nodiscard]] std::pair<val_t, std::optional<uint32_t>> parse_heuristic(char const *value) {
    std::optional<uint32_t> thread = std::nullopt;
    char const *comma = find_str(value, ',');
    if (*comma != '\0') {
        thread = parse_num<uint32_t, 0, MAX_THREADS - 1>(comma + 1); // NOLINT
    }

    if (std::strncmp(value, "none", comma - value) == 0) {
        return {static_cast<val_t>(Clingcon::Heuristic::None), thread};
    }
    if (std::strncmp(value, "max-chain", comma - value) == 0) {
        return {static_cast<val_t>(Clingcon::Heuristic::MaxChain), thread};
    }
    throw std::invalid_argument("invalid argument");
}

[[nodiscard]] std::function<bool (const char *)> parser_bool_thread(clingcon_theory &theory, Target target) {
    return [&theory, target](char const *value) {
        auto [val, thread] = parse_bool_thread(value);
        return theory.deferred.emplace(std::pair(target, thread), val).second;
    };
}

[[nodiscard]] std::function<bool (const char *)> parser_sign_value(clingcon_theory &theory, Target target) {
    return [&theory, target](char const *value) {
        auto [val, thread] = parse_sign_value(value);
        return theory.deferred.emplace(std::pair(target, thread), val).second;
    };
}

[[nodiscard]] std::function<bool (const char *)> parser_heuristic(clingcon_theory &theory) {
    return [&theory](char const *value) {
        auto [val, thread] = parse_heuristic(value);
        return theory.deferred.emplace(std::pair(Target::Heuristic, thread), val).second;
    };
}

} // namespace

extern "C" bool clingcon_create(clingcon_theory_t **theory) {
    CLINGCON_TRY {
        *theory = new clingcon_theory(); // NOLINT
    }
    CLINGCON_CATCH;
}

extern "C" bool clingcon_register(clingcon_theory_t *theory, clingo_control_t* control) {
    // Note: The decide function is passed here for performance reasons.
    auto &config = theory->propagator.config();
    bool has_heuristic = config.default_solver_config.heuristic != Clingcon::Heuristic::None;
    for (auto &sconfig : config.solver_configs) {
        if (has_heuristic) { break; }
        has_heuristic = sconfig.heuristic != Clingcon::Heuristic::None;
    }

    static clingo_propagator_t propagator = { init<Clingcon::Propagator>, propagate<Clingcon::Propagator>,
                                              undo<Clingcon::Propagator>, check<Clingcon::Propagator>,
                                              has_heuristic ? decide<Clingcon::Propagator> : nullptr };
    static clingo_propagator_t dlpropagator = { nullptr,
                                                propagate<ClingoDL::DifferenceLogicPropagator>,
                                                undo<ClingoDL::DifferenceLogicPropagator>,
                                                check<ClingoDL::DifferenceLogicPropagator>,
                                                nullptr };
    return
        clingo_control_add(control, "base", nullptr, 0, Clingcon::THEORY) &&
        clingo_control_register_propagator(control, &dlpropagator, &theory->dlpropagator, false) &&
        clingo_control_register_propagator(control, &propagator, &theory->propagator, false);
}

extern "C" bool clingcon_rewrite_statement(clingcon_theory_t *theory, clingo_ast_statement_t const *stm, clingcon_rewrite_callback_t add, void *data) {
    CLINGCON_TRY {
        Clingo::StatementCallback cb = [&](Clingo::AST::Statement &&stm) {
            transform(std::move(stm), [add, data](Clingo::AST::Statement &&stm){
                Clingo::AST::Detail::ASTToC visitor;
                auto x = stm.data.accept(visitor);
                x.location = stm.location;
                handle_error(add(&x, data));
            }, theory->shift_constraints);
        };
        Clingo::AST::Detail::convStatement(stm, cb);
    }
    CLINGCON_CATCH;
}

extern "C" bool clingcon_prepare(clingcon_theory_t *theory, clingo_control_t* control) {
    static_cast<void>(theory);
    static_cast<void>(control);
    // Note: There is nothing todo.
    return true;
}

extern "C" bool clingcon_destroy(clingcon_theory_t *theory) {
    delete theory; // NOLINT
    return true;
}

extern "C" bool clingcon_configure(clingcon_theory_t *theory, char const *key, char const *value) {
    CLINGCON_TRY {
        auto config = theory->propagator.config();
        // translation
        if (std::strcmp(key, "shift-constraints") == 0) {
            theory->shift_constraints = parse_bool(value);
        }
        else if (std::strcmp(key, "sort-constraints") == 0) {
            config.sort_constraints = parse_bool(value);
        }
        else if (std::strcmp(key, "translate-clauses") == 0) {
            config.clause_limit = parse_num<uint32_t>(value);
        }
        else if (std::strcmp(key, "literals-only") == 0) {
            config.literals_only = parse_bool(value);
        }
        else if (std::strcmp(key, "translate-pb") == 0) {
            config.weight_constraint_limit = parse_num<uint32_t>(value);
        }
        else if (std::strcmp(key, "translate-distinct") == 0) {
            config.distinct_limit = parse_num<uint32_t>(value);
        }
        else if (std::strcmp(key, "translate-opt") == 0) {
            config.translate_minimize = parse_bool(value);
        }
        else if (std::strcmp(key, "add-order-clauses") == 0) {
            config.add_order_clauses = parse_bool(value);
        }
        // hidden/debug
        else if (std::strcmp(key, "min-int") == 0) {
            config.min_int = parse_num<val_t, MIN_VAL, MAX_VAL>(value);
        }
        else if (std::strcmp(key, "max-int") == 0) {
            config.max_int = parse_num<val_t, MIN_VAL, MAX_VAL>(value);
        }
        else if (std::strcmp(key, "check-solution") == 0) {
            config.check_solution = parse_bool(value);
        }
        else if (std::strcmp(key, "check-state") == 0) {
            config.check_state = parse_bool(value);
        }
        // propagation
        else if (std::strcmp(key, "order-heuristic") == 0) {
            set_value(Target::Heuristic, config, parse_heuristic(value));
        }
        else if (std::strcmp(key, "sign-value") == 0) {
            set_value(Target::SignValue, config, parse_sign_value(value));
        }
        else if (std::strcmp(key, "refine-reasons") == 0) {
            set_value(Target::RefineReasons, config, parse_bool_thread(value));
        }
        else if (std::strcmp(key, "refine-introduce") == 0) {
            set_value(Target::RefineIntroduce, config, parse_bool_thread(value));
        }
        else if (std::strcmp(key, "propagate-chain") == 0) {
            set_value(Target::PropagateChain, config, parse_bool_thread(value));
        }
        else if (std::strcmp(key, "split-all") == 0) {
            set_value(Target::SplitAll, config, parse_bool_thread(value));
        }
    }
    CLINGCON_CATCH;
}

extern "C" bool clingcon_register_options(clingcon_theory_t *theory, clingo_options_t* options) {
    CLINGCON_TRY {
        char const *group = "CSP Options";
        auto &config = theory->propagator.config();
        Clingo::ClingoOptions opts{options, theory->parsers};

        // translation
        opts.add_flag(
            group, "shift-constraints",
            format("Shift constraints into head of integrity constraints [", flag_str(theory->shift_constraints), "]").c_str(),
            theory->shift_constraints);
        opts.add_flag(
            group, "sort-constraints",
            format("Sort constraint elements [", flag_str(config.sort_constraints), "]").c_str(),
            config.sort_constraints);
        opts.add(
            group, "translate-clauses",
            format("Restrict translation to <n> clauses per constraint [", config.clause_limit, "]").c_str(),
            parser_num<uint32_t>(config.clause_limit), false, "<n>");
        opts.add_flag(
            group, "literals-only",
            format("Only create literals during translation but no clauses [", flag_str(config.literals_only), "]").c_str(),
            config.literals_only);
        opts.add(
            group, "translate-pb",
            format("Restrict translation to <n> literals per pb constraint [", config.weight_constraint_limit, "]").c_str(),
            parser_num<uint32_t>(config.weight_constraint_limit), false, "<n>");
        opts.add(
            group, "translate-distinct",
            format("Restrict translation of distinct constraints <n> pb constraints [", config.distinct_limit, "]").c_str(),
            parser_num<uint32_t>(config.distinct_limit), false, "<n>");
        opts.add_flag(
            group, "translate-opt",
            format("Translate minimize constraint into clasp's minimize constraint [", flag_str(config.translate_minimize), "]").c_str(),
            config.translate_minimize);
        opts.add_flag(
            group, "add-order-clauses",
            format("Add binary clauses for order literals after translation [", flag_str(config.add_order_clauses), "]").c_str(),
            config.add_order_clauses);

        // propagation
        opts.add(
            group, "order-heuristic",
            format(
                "Make the decision heuristic aware of order literls [", heuristic_str(config.default_solver_config.heuristic), "]\n"
                "      <arg>: {none,max-chain}[,<i>]\n"
                "      none     : use clasp's heuristic\n"
                "      max-chain: assign chains of literals\n"
                "      <i>      : Only enable for thread <i>").c_str(),
            parser_heuristic(*theory), true);
        opts.add(
            group, "sign-value",
            format(
                "Configure the sign of order literals [", config.default_solver_config.sign_value, "]\n"
                "      <arg>: {<n>|+|-}[,<i>]\n"
                "      <n>: negative if its value is greater or equal to <n>\n"
                "      <+>: always positive\n"
                "      <->: always negative\n"
                "      <i>: Only enable for thread <i>").c_str(),
            parser_sign_value(*theory, Target::SignValue), true);
        opts.add(
            group, "refine-reasons",
            format(
                "Refine reasons during propagation [", flag_str(config.default_solver_config.refine_reasons), "]\n"
                "      <arg>: {yes|no}[,<i>]\n"
                "      <i>: Only enable for thread <i>").c_str(),
            parser_bool_thread(*theory, Target::RefineReasons), true);
        opts.add(
            group, "refine-introduce",
            format(
                "Introduce order literals when generating reasons [", flag_str(config.default_solver_config.refine_introduce), "]\n"
                "      <arg>: {yes|no}[,<i>]\n"
                "      <i>: Only enable for thread <i>").c_str(),
            parser_bool_thread(*theory, Target::RefineIntroduce), true);
        opts.add(
            group, "propagate-chain",
            format(
                "Use closest order literal as reason [", flag_str(config.default_solver_config.propagate_chain), "]\n"
                "      <arg>: {yes|no}[,<i>]\n"
                "      <i>: Only enable for thread <i>").c_str(),
            parser_bool_thread(*theory, Target::PropagateChain), true);
        opts.add(
            group, "split-all",
            format(
                "Split all domains on total assignment [", flag_str(config.default_solver_config.split_all), "]\n"
                "      <arg>: {yes|no}[,<i>]\n"
                "      <i>: Only enable for thread <i>").c_str(),
            parser_bool_thread(*theory, Target::SplitAll), true);

        // hidden/debug
        opts.add(
            group, "min-int,@2",
            format("Set minimum integer [", config.min_int, "]").c_str(),
            parser_num<val_t, MIN_VAL, MAX_VAL>(config.min_int), false, "<i>");
        opts.add(
            group, "max-int,@2",
            format("Set maximum integer [", config.max_int, "]").c_str(),
            parser_num<val_t, MIN_VAL, MAX_VAL>(config.max_int), false, "<i>");
        opts.add_flag(
            group, "check-solution,@2",
            format("Verify solutions [", flag_str(config.check_solution), "]").c_str(),
            config.check_solution);
        opts.add_flag(
            group, "check-state,@2",
            format("Check state invariants [", flag_str(config.check_state), "]").c_str(),
            config.check_state);
            
//        group = "Clingo.DL Options";
//        opts.add(
//            group, "add-mutexes",
//            "Add mutexes in a preprocessing step [0]\n"
//            "      <arg>   : <max>[,<cut>]\n"
//            "      <max>   : Maximum size of mutexes to add\n"
//            "      <cut>   : Limit costs to calculate mutexes\n",
//            &parse_mutex, &theory->dlconf, true, "<arg>"));
    }
    CLINGCON_CATCH;
}

extern "C" bool clingcon_validate_options(clingcon_theory_t *theory) {
    CLINGCON_TRY {
        auto &config = theory->propagator.config();

        for (auto has_value : {false, true}) {
            for (auto [target_thread, value] : theory->deferred) {
                auto [target, thread] = target_thread;
                if (has_value == thread.has_value()) {
                    set_value(target, config, {value, thread});
                }
            }
        }
        theory->deferred.clear();

        if (config.min_int > config.max_int) {
            throw std::runtime_error("min-int must be smaller than or equal to max-int");
        }
    }
    CLINGCON_CATCH;
}

extern "C" bool clingcon_on_model(clingcon_theory_t *theory, clingo_model_t* model) {
    CLINGCON_TRY {
        Clingo::Model m{model};
        theory->propagator.on_model(m);
    }
    CLINGCON_CATCH;
}

extern "C" bool clingcon_lookup_symbol(clingcon_theory_t *theory, clingo_symbol_t symbol, size_t *index) {
    if (auto var = theory->propagator.get_index(Clingo::Symbol{symbol}); var.has_value()) {
        *index = *var + 1;
        return true;
    }
    return false;
}

extern "C" clingo_symbol_t clingcon_get_symbol(clingcon_theory_t *theory, size_t index) {
    auto sym = theory->propagator.get_symbol(index - 1);
    assert(sym.has_value());
    return sym->to_c();
}

extern "C" void clingcon_assignment_begin(clingcon_theory_t *theory, uint32_t thread_id, size_t *index) {
    static_cast<void>(theory);
    static_cast<void>(thread_id);
    *index = 0;
}

extern "C" bool clingcon_assignment_next(clingcon_theory_t *theory, uint32_t thread_id, size_t *index) {
    static_cast<void>(thread_id);
    auto const &map = theory->propagator.var_map();
    auto it = map.lower_bound(*index);
    if (it != map.end()) {
        *index = *index + 1;
        return true;
    }
    return false;
}

extern "C" bool clingcon_assignment_has_value(clingcon_theory_t *theory, uint32_t thread_id, size_t index) {
    static_cast<void>(thread_id);
    return theory->propagator.get_symbol(index - 1).has_value();
}

extern "C" void clingcon_assignment_get_value(clingcon_theory_t *theory, uint32_t thread_id, size_t index, clingcon_value_t *value) {
    value->type = clingcon_value_type_int; // NOLINT
    value->int_number = theory->propagator.get_value(index - 1, thread_id); // NOLINT
}

extern "C" bool clingcon_on_statistics(clingcon_theory_t *theory, clingo_statistics_t* step, clingo_statistics_t* accu) {
    uint64_t step_root, accu_root; // NOLINT
    if (!clingo_statistics_root(step, &step_root) || !clingo_statistics_root(accu, &accu_root)) {
        return false;
    }
    CLINGCON_TRY {
        Clingo::UserStatistics step_stats{step, step_root};
        Clingo::UserStatistics accu_stats{accu, accu_root};
        theory->propagator.on_statistics(step_stats, accu_stats);
        theory->on_dl_statistics(step_stats, accu_stats);
    }
    CLINGCON_CATCH;
}
