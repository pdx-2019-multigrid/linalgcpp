#include "argparser.hpp"

namespace linalgcpp
{

ArgParser::ArgParser(int argc, const char* const* argv)
{
    const std::string help_string = "--help";

    std::vector<std::string> entries;

    for (int i = 1; i < argc; ++i)
    {
        if (help_string.compare(argv[i]) == 0)
        {
            need_help_ = true;
            continue;
        };

        entries.push_back(argv[i]);
        assert(entries.back().size() > 0);
    }

    for (auto iter = entries.begin(); iter != entries.end(); /* empty */)
    {
        assert(iter->size() > 0);
        assert((*iter)[0] == '-');

        if ((iter + 1) != entries.end() && (*(iter + 1))[0] != '-')
        {
            if (values_.find(*iter) == values_.end())
            {
                values_[*iter] = *(iter + 1);
            }
            else
            {
                errors_.push_back(*iter + " entered multiple times!");
            }

            iter += 2;;
        }
        else
        {
            if (values_.find(*iter) == values_.end())
            {
                values_[*iter] = "";
            }
            else
            {
                errors_.push_back(*iter + " entered multiple times!");
            }

            iter += 1;
        }
    }
}

void ArgParser::ShowErrors(std::ostream& out) const
{
    if (errors_.size())
    {
        out << "Parse Errors:\n";
    }

    for (auto& err : errors_)
    {
        out << "  " << err << "\n";
    }

    out.flush();
}

void ArgParser::ShowHelp(std::ostream& out) const
{
    out << "Program Help:\n";

    for (auto& option : help_)
    {
        out << "  " << option << "\n";
    }

    out.flush();
}

void ArgParser::ShowOptions(std::ostream& out) const
{
    out << "Program Options:\n";

    for (auto& option : seen_values_)
    {
        out << "  " << option.first << "\t" << option.second << "\n";
    }

    out.flush();
}

template <>
void ArgParser::Parse(bool& arg, const std::string& flag, const std::string& help) const
{
    help_.push_back(flag + "\t" + help);

    if (values_.find(flag) != values_.end())
    {
        arg = !arg;
    }

    if (seen_values_.find(flag) == seen_values_.end())
    {
        std::stringstream ss;
        ss << std::boolalpha << arg;

        seen_values_[flag] = ss.str();
    }
    else
    {
        errors_.push_back(flag + " requested to be parsed multiple times!");
    }
}

bool ArgParser::IsGood() const
{
    return (errors_.size() == 0 && !need_help_);
}

} // namespace linalgcpp

