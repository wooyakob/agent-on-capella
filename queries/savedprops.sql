# Bucket Profiles, Scope Buyers
# Return saved properties for buyer

SELECT sp.*
FROM `2025` c
UNNEST c.saved_properties AS sp
LIMIT 5;