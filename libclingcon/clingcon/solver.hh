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

#ifndef CLINGCON_SOLVER_H
#define CLINGCON_SOLVER_H

#include <clingcon/base.hh>
#include <map>

//! @file clingcon/solver.hh
//! This module implements a CSP solver for thread-specific propagation and
//! defines interfaces for constraints.
//!
//! @author Roland Kaminski

namespace Clingcon {

class Solver;
class AbstractConstraint;
class AbstractConstraintState;
using UniqueConstraint = std::unique_ptr<AbstractConstraint>;
using UniqueConstraintState = std::unique_ptr<AbstractConstraintState>;

//! Base class of all constraints.
class AbstractConstraint {
public:
    AbstractConstraint() = default;
    AbstractConstraint(AbstractConstraint const &) = delete;
    AbstractConstraint(AbstractConstraint &&) = delete;
    AbstractConstraint &operator=(AbstractConstraint const &) = delete;
    AbstractConstraint &operator=(AbstractConstraint &&) = delete;
    virtual ~AbstractConstraint() = default;

    //! Create thread specific state for the constraint.
    virtual UniqueConstraintState create_state() = 0;
};


//! Abstract class to capture the state of constraints.
class AbstractConstraintState {
public:
    AbstractConstraintState() = default;
    AbstractConstraintState(AbstractConstraintState const &) = delete;
    AbstractConstraintState(AbstractConstraintState &&) = delete;
    AbstractConstraintState &operator=(AbstractConstraintState const &) = delete;
    AbstractConstraintState &operator=(AbstractConstraintState &&) = delete;
    virtual ~AbstractConstraintState() = default;

    //! Get a refernce to the level on which the constraint became inactive.
    virtual level_t &inactive_level() = 0;
    //! Attach the constraint to a solver.
    virtual void attach(Solver &solver) = 0;
    //! Detach the constraint from a solver.
    virtual void detach(Solver &solver) = 0;
    //! Translate a constraint to simpler constraints.
    virtual bool translate(Solver &solver, AbstractClauseCreator &cc, Config &config, std::vector<UniqueConstraintState> &added) = 0;
    //! Inform the solver about updated bounds of a variable.
    //!
    //! Value i depends on the value passed when registering the watch and diff
    //! is the change to the bound of the watched variable.
    virtual void update(val_t i, val_t diff) = 0;
    //! Similar to update but when the bound of a variable is backtracked.
    virtual void undo(val_t i, val_t diff) = 0;
    //! Prepagates the constraint.
    virtual bool propagate(Solver &solver, AbstractClauseCreator &cc, SolverConfig &config, bool check_state) = 0;
    //! Check if the solver meets the state invariants.
    virtual bool check_full(Solver &solver);

    //! Returns true if the constraint is marked inactive.
    bool marked_inactive() {
        return inactive_level() > 0;
    }

    //! Mark a constraint inactive on the given level.
    void mark_inactive(level_t level) {
        assert(!marked_inactive());
        inactive_level() = level + 1;
    }

    //! Mark a constraint active.
    void mark_active() {
        inactive_level() = 0;
    }

    //! A constraint is removable if it has been marked inactive on a lower
    //! level.
    bool removable(level_t level) {
        return marked_inactive() && inactive_level() <= level;
    }
};

//! Class to facilitate handling order literals associated with an integer
//! variable.
//!
//! The class maintains a stack of lower and upper bounds, which initially
//! contain the smallest and largest allowed integer. These stacks must always
//! contain at least one value.
class VarState {
public:
    using OrderLiterals = std::map<val_t, lit_t>;                  //!< Container to store order literals.
    using Iterator = OrderLiterals::const_iterator;                //!< Iterator over order literals.
    using ReverseIterator = OrderLiterals::const_reverse_iterator; //!< Reverse iterator over order literals.

    //! Create an initial state for the given variable.
    //!
    //! Initially, the state should have  a lower bound of `Config::min_int`
    //! and an upper bound of `Config::max_int` and is associated with no
    //! variables.
    VarState(var_t var, val_t lower_bound, val_t upper_bound)
    : var_{var}
    , lower_bound_{lower_bound}
    , upper_bound_{upper_bound} {
    }
    VarState() = delete;
    VarState(VarState const &) = default;
    VarState(VarState &&) noexcept = default;
    VarState &operator=(VarState const &) = default;
    VarState &operator=(VarState &&) noexcept = default;
    ~VarState() = default;

    //! Remove all literals associated with this state.
    void reset(val_t min_int, val_t max_int) {
        lower_bound_ = min_int;
        upper_bound_ = max_int;
        lower_bound_stack_.clear();
        upper_bound_stack_.clear();
        literals_.clear();
    }

    //! @name Functions for Lower Bounds
    //! @{

    //! Get current lower bound.
    [[nodiscard]] val_t lower_bound() const {
        return lower_bound_;
    }
    //! Set new lower bound.
    void lower_bound(val_t lower_bound) {
        assert(lower_bound >= lower_bound_);
        lower_bound_ = lower_bound;
    }
    //! Push the current lower bound on the stack.
    void push_lower(level_t level) {
        lower_bound_stack_.emplace_back(level, lower_bound_);
    }
    //! Check if the given level has already been pushed on the stack.
    [[nodiscard]] bool pushed_lower(level_t level) const {
        return !lower_bound_stack_.empty() && lower_bound_stack_.back().first == level;
    }
    //! Pop and restore last lower bound from the stack.
    void pop_lower() {
        assert(!lower_bound_stack_.empty());
        lower_bound_ = lower_bound_stack_.back().second;
        lower_bound_stack_.pop_back();
    }
    //! Get the smallest possible value the variable can take.
    [[nodiscard]] val_t min_bound() const {
        return lower_bound_stack_.empty() ? lower_bound_ : lower_bound_stack_.front().second;
    }

    //! @}

    //! @name Functions for Upper Bounds
    //! @{

    //! Get current upper bound.
    [[nodiscard]] val_t upper_bound() const {
        return lower_bound_;
    }
    //! Set new upper bound.
    void upper_bound(val_t upper_bound) {
        assert(upper_bound <= upper_bound_);
        upper_bound_ = upper_bound;
    }
    //! Push the current upper bound on the stack.
    void push_upper(level_t level) {
        upper_bound_stack_.emplace_back(level, upper_bound_);
    }
    //! Check if the given level has already been pushed on the stack.
    [[nodiscard]] bool pushed_upper(level_t level) const {
        return !upper_bound_stack_.empty() && upper_bound_stack_.back().first == level;
    }
    //! Pop and restore last upper bound from the stack.
    void pop_upper() {
        assert(!upper_bound_stack_.empty());
        upper_bound_ = upper_bound_stack_.back().second;
        upper_bound_stack_.pop_back();
    }
    //! Get the smallest possible value the variable can take.
    [[nodiscard]] val_t max_bound() const {
        return upper_bound_stack_.empty() ? upper_bound_ : upper_bound_stack_.front().second;
    }

    //! @}

    //! @name Functions for Values and Literals
    //! @{

    //! Get the variable index of the state.
    [[nodiscard]] var_t var() const {
        return var_;
    }

    //! Determine if the variable is assigned, i.e., the current lower bound
    //! equals the current upper bound.
    [[nodiscard]] bool is_assigned() const {
        return lower_bound_ == upper_bound_;
    }

    //! Determine if the given value is associated with an order literal.
    [[nodiscard]] bool has_literal(val_t value) const {
        return literals_.find(value) != literals_.end();
    }
    //! Get the literal associated with the value.
    [[nodiscard]] std::optional<lit_t> get_literal(val_t value) {
        auto it = literals_.find(value);
        if (it != literals_.end()) {
            return it->second;
        }
        return {};
    }

    //! Set the literal of the given value.
    void set_literal(val_t value, lit_t lit) {
        literals_[value] = lit;
    }
    //! Unset the literal of the given value.
    void unset_literal(val_t value) {
        literals_.erase(value);
    }

    //! @}

    //! @name Functions for Traversing Order Literals
    //! @{

    //! Return an iterator to the first order literal.
    [[nodiscard]] Iterator begin() const {
        return literals_.begin();
    }
    //! Return an iterator after the last order literal.
    [[nodiscard]] Iterator end() const {
        return literals_.begin();
    }

    //! Return a reverse iterator to the last order literal.
    [[nodiscard]] ReverseIterator rbegin() const {
        return literals_.rbegin();
    }
    //! Return a reverse iterator before the first order literal.
    [[nodiscard]] Iterator rend() const {
        return literals_.begin();
    }

    //! Return a reverse iterator to the first order literal with a value less
    //! than the given value.
    ReverseIterator lit_lt(val_t value) {
        return ReverseIterator{literals_.lower_bound(value)};
    }
    //! Return a reverse iterator to the first order literal with a value less
    //! than or equal to the given value.
    ReverseIterator lit_le(val_t value) {
        return ReverseIterator{literals_.upper_bound(value)};
    }

    //! Return an iterator to the first order literal with a value greater than
    //! the given value.
    Iterator lit_gt(val_t value) {
        return literals_.upper_bound(value);
    }
    //! Return an iterator to the first order literal with a value greater than
    //! or equal to the given value.
    Iterator lit_ge(val_t value) {
        return literals_.lower_bound(value);
    }

    //! @}

private:
    using BoundStack = std::vector<std::pair<level_t, val_t>>;

    var_t var_;                    //!< variable associated with the state
    val_t lower_bound_;            //!< current lower bound of the variable
    val_t upper_bound_;            //!< current upper bound of the variable
    BoundStack lower_bound_stack_; //!< lower bounds of lower levels
    BoundStack upper_bound_stack_; //!< upper bounds of lower levels
    OrderLiterals literals_;       //!< map from values to literals
};

class Solver {
    struct Level;

public:
    Solver() = delete;
    Solver(Solver const &) = delete;
    Solver(Solver &&) = delete;
    Solver& operator=(Solver const &) = delete;
    Solver& operator=(Solver &&) = delete;
    ~Solver();

private:
    std::vector<Level> levels_;
};

} // namespace Clingcon

#endif // CLINGCON_SOLVER_H
