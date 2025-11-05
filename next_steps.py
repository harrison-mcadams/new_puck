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
#   - scraped season data from the API -- from the repo's main folder,
#   create a subfolder 'data' to store cached data files. within that,
#   we'll further nest events. within this subfolder, for each season,
#   i'd like to:
#       - within parse, write a function called _scrape. it will take year as
#       input, use the get_season function from nhl_api, and then for each
#       game across the league within that season,
#       call nhl.api.get_game_data. it should then save out a single csv file
#       of the concatenated game feeds from all games in that season. do not
#       touch the other parts of parse right now.
#   - elaborated season data
#   - xg model
#   - website shortcuts: team list, game list, player list
#   - add remove shoot out attempts from game plotting

# Model improvements:
#   - add more features to xg model. next up shot type