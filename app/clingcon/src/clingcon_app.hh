// {{{ GPL License

// This file is part of gringo - a grounder for logic programs.
// Copyright (C) 2013  Benjamin Kaufmann
// Copyright (C) 2013  Roland Kaminski

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// }}}

#ifndef CLINGO_CLINGCONAPP_HH
#define CLINGO_CLINGCONAPP_HH

#include "version.h"
#include <clasp_app.h>
#include <clingo/clingocontrol.hh>
#include <order/config.h>

// Standalone clingo application.
class ClingconApp : public Clasp::Cli::ClaspAppBase {
    using ClaspOutput = Clasp::Cli::Output;
    using ProblemType = Clasp::ProblemType;
    using BaseType    = Clasp::Cli::ClaspAppBase;
    enum class ConfigUpdate { KEEP, REPLACE };
public:
    ClingconApp();
    const char* getName()    const override { return "clingcon"; }
    const char* getVersion() const override { return CLINGCON_VERSION; }
    const char* getUsage()   const override { return "[number] [options] [files]"; }

    void shutdown() override;
protected:
    enum Mode { mode_clingo = 0, mode_clasp = 1, mode_gringo = 2 };
    void        initOptions(Potassco::ProgramOptions::OptionContext& root) override;
    void        validateOptions(const Potassco::ProgramOptions::OptionContext& root, const Potassco::ProgramOptions::ParsedOptions& parsed, const Potassco::ProgramOptions::ParsedValues& vals) override;

    ProblemType getProblemType() override;
    void        run(Clasp::ClaspFacade& clasp) override;
    ClaspOutput* createOutput(ProblemType f) override;
    void        printHelp(const Potassco::ProgramOptions::OptionContext& root) override;
    void        printVersion() override;

    // -------------------------------------------------------------------------------------------
    // Event handler
    void onEvent(const Clasp::Event& ev) override;
    bool onModel(const Clasp::Solver& s, const Clasp::Model& m) override;
    // -------------------------------------------------------------------------------------------
private:
    ClingconApp(const ClingconApp&);
    ClingconApp& operator=(const ClingconApp&);
    Gringo::ClingoOptions grOpts_;
    Mode mode_;
    std::unique_ptr<Gringo::ClingoControl> grd;
    order::Config conf_;
};

#endif // CLINGO_CLINGCONAPP_HH
