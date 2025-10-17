CREATE PRIMARY INDEX `buyers_2025_pidx` ON `profiles`.`buyers`.`2025`;

CREATE PRIMARY INDEX `primary` ON `profiles`.`tours`.`2025`;

CREATE INDEX ix_tours_buyer_created ON `profiles`.`tours`.`2025`(LOWER(buyer_name), created_at);