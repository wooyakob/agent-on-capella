SELECT b.*
FROM profiles.`buyers`.`2025` AS b
WHERE b.buyer IS NOT MISSING
ORDER BY b.buyer;