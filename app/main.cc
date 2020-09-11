#include <clingo.hh>
#include <clingcon.h>
#include <clingo-dl.h>
#include <sstream>
#include <fstream>
#include <optional>

#ifdef CLINGCON_PROFILE
#include <gperftools/profiler.h>
#endif

using Clingo::Detail::handle_error;


class Rewriter {
public:
    Rewriter(clingcon_theory_t *csptheory, clingo_program_builder_t *builder)
    : csptheory_{csptheory}
    , builder_{builder} {
    }

    void load(char const *file) {
        std::string program;
        if (strcmp(file, "-") == 0) {
            program.assign(std::istreambuf_iterator<char>{std::cin}, std::istreambuf_iterator<char>{});
        }
        else {
            std::ifstream ifs{file};
            if (!ifs.is_open()) {
                std::ostringstream oss;
                oss << "could not open file: " << file;
                throw std::runtime_error(oss.str());
            }
            program.assign(std::istreambuf_iterator<char>{ifs}, std::istreambuf_iterator<char>{});
        }

        // TODO: parsing from file would be nice
        handle_error(clingo_parse_program(program.c_str(), rewrite_, this, nullptr, nullptr, 0));
    }

private:
    static bool add_(clingo_ast_statement_t const *stm, void *data) {
        auto *self = static_cast<Rewriter*>(data);
        return clingo_program_builder_add(self->builder_, stm);
    }

    static bool rewrite_(clingo_ast_statement_t const *stm, void *data) {
        auto *self = static_cast<Rewriter*>(data);
        return clingcon_rewrite_statement(self->csptheory_, stm, add_, self);
    }

    clingcon_theory_t *csptheory_;
    clingo_program_builder_t *builder_;
};


class ClingconApp final : public Clingo::Application, private Clingo::SolveEventHandler {
public:
    ClingconApp() {
        handle_error(clingcon_create(&csptheory_));
        handle_error(clingodl_create(&dltheory_));
    }

    ClingconApp(ClingconApp const &) = delete;
    ClingconApp(ClingconApp &&) = delete;
    ClingconApp &operator=(ClingconApp const &) = delete;
    ClingconApp &operator=(ClingconApp &&) = delete;

    ~ClingconApp() override {
        if (csptheory_ != nullptr) {
            clingcon_destroy(csptheory_);
        }
        if (dltheory_ != nullptr) {
            clingodl_destroy(dltheory_);
        }
    }

    [[nodiscard]] char const *program_name() const noexcept override {
        return "clingcon";
    }

    [[nodiscard]] char const *version() const noexcept override {
        return CLINGCON_VERSION;
    }

    void register_options(Clingo::ClingoOptions &options) override {
        handle_error(clingcon_register_options(csptheory_, options.to_c()));
        handle_error(clingodl_register_options(dltheory_, options.to_c()));
    }

    void validate_options() override {
        handle_error(clingcon_validate_options(csptheory_));
        handle_error(clingodl_validate_options(dltheory_));
    }

    bool on_model(Clingo::Model &model) override {
        handle_error(clingcon_on_model(csptheory_, model.to_c()));
        return true;
    }

    void print_model(Clingo::Model const &model, std::function<void()> default_printer) noexcept override {
        static_cast<void>(default_printer);
        try {
            // print model
            bool comma = false;
            auto symbols = model.symbols(Clingo::ShowType::Shown);
            symvec_.assign(symbols.begin(), symbols.end());
            std::sort(symbols.begin(), symbols.end());
            for (auto &sym : symbols) {
                std::cout << (comma ? " " : "") <<  sym;
                comma = true;
            }
            std::cout << "\n";

            // print assignment
            comma = false;
            symbols = model.symbols(Clingo::ShowType::Theory);
            symvec_.assign(symbols.begin(), symbols.end());
            std::sort(symbols.begin(), symbols.end());
            char const *cost = nullptr;
            std::cout << "Assignment:\n";
            for (auto &sym : symbols) {
                if (sym.match("__csp", 2)) {
                    auto arguments = sym.arguments();
                    std::cout << (comma ? " " : "") <<  arguments[0] << "=" << arguments[1];
                    comma = true;
                }
                else if (sym.match("__csp_cost", 1)) {
                    auto arguments = sym.arguments();
                    if (arguments[0].type() == Clingo::SymbolType::String) {
                        cost = arguments[0].string();
                    }
                }
            }
            std::cout << "\n";

            // print cost
            if (cost != nullptr) {
                std::cout << "Cost: " << cost << "\n";
            }

            std::cerr.flush();
        }
        catch(...) {
            std::terminate();
        }

    }
    void on_statistics(Clingo::UserStatistics step, Clingo::UserStatistics accu) override {
        handle_error(clingcon_on_statistics(csptheory_, step.to_c(), accu.to_c()));
        handle_error(clingodl_on_statistics(dltheory_, step.to_c(), accu.to_c()));
    }

    void main(Clingo::Control &control, Clingo::StringSpan files) override { // NOLINT(bugprone-exception-escape)
        handle_error(clingodl_register(dltheory_, control.to_c()));
        handle_error(clingcon_register(csptheory_, control.to_c()));

        parse_(control, files);
        control.ground({{"base", {}}});
        handle_error(clingodl_prepare(dltheory_, control.to_c()));
        handle_error(clingcon_prepare(csptheory_, control.to_c()));

#ifdef CLINGCON_PROFILE
        ProfilerStart("clingcon.solve.prof");
#endif
        control.solve(Clingo::SymbolicLiteralSpan{}, this, false, false).get();
#ifdef CLINGCON_PROFILE
        ProfilerStop();
#endif
    }

private:
    void parse_(Clingo::Control &control, Clingo::StringSpan files) {
        control.with_builder([&](Clingo::ProgramBuilder &builder) {
            Rewriter rewriter{csptheory_, builder.to_c()};
            for (auto const &file : files) {
                rewriter.load(file);
            }
            if (files.empty()) {
                rewriter.load("-");
            }
        });
    }

    clingcon_theory_t *csptheory_{nullptr};
    clingodl_theory_t *dltheory_{nullptr};
    std::vector<Clingo::Symbol> symvec_;
};


int main(int argc, char *argv[]) {
    ClingconApp app;
    return Clingo::clingo_main(app, {argv + 1, static_cast<size_t>(argc - 1)});
}
