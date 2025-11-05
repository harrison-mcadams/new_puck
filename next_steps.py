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
#   - xg model
#   - website shortcuts: team list, game list, player list
#   - add remove shoot out attempts from game plotting

# Model improvements:
#   - add more features to xg model. next up shot type