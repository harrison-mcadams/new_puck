## This document outlines plans for next steps.

# General principles:
#   - Prioritize direct approaches. Previous iterations of the code have had
#   a lot of extra optionality to handle edge cases. For example, there is a
#   lot of code that handles different potential field names for a variable
#   of interest. While this is useful for robustness, it adds complexity. I
#   would like more direct control and assume a standard format.
#$   - Prioritize clarity over cleverness. The code should be easy to read
#   and understand, even if that means being more verbose or less optimized.

# Work on cacheing.
# Things to cache:
#   - website shortcuts: team list, game list, player list
#   - add remove shoot out attempts from game plotting

# Model improvements:
#   - add more features to xg model. next up shot type

# Refactoring:
#   - in parse.py (the routine that that works on dfs), create a routine
#   called filter. you should be able to filter the df on any number of
#   cases, but the obvious use case is 'team'. this is currently working in
#   analyze.py but should be generalized and moved to parse.py
#   - eventually rename plot.plot_events to plot._events

# Analysis
#   - Make analysis.py which has a routine that looks at relative xgs for a
#   given team in the offensive and defensive zones
#   - plot._game function. it will do the filtering of the game df, and then
#   pass the clean event log to plot_events.


# Desired behavior, considering 5v4:
# - when I'm working on shifts data:
    # - for league:
        # - only '5v4' applies. but '4v5' shots will take place within the
# same time periods.
    # - for team:
# - when i'm working on xgs map:
    # - for league:
        # - only '5v4' applies
        # - all power play offensive shots oriented to the lef
        # - home/away will remain divided, can just be summed up
# - for team:
        # - only '5v4' applies
        # - all power play offensive shots oriented to the left -- inputted
# team_id_of_interest matches 'team_id' for plays in '5v4' game state
        # - all '5v4' from 'team_id' not matching 'team of interest',
# with shots oriented to the left. This is the Flyers penalty kill
