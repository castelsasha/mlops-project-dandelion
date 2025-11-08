-- scripts/sql/seed_plants_data_mysql.sql
-- Purpose:
--   Insert a few rows WITH url_s3 = NULL so that the pull_and_push ETL
--   will pick them up automatically and upload them to S3.
--
-- This file is mainly here so a new developer can test the pipeline
-- without manually inserting data in MySQL.

INSERT INTO plants_data (url_source, url_s3, label) VALUES
('https://images.pexels.com/photos/414660/pexels-photo-414660.jpeg', NULL, 'dandelion'),
('https://images.pexels.com/photos/259280/pexels-photo-259280.jpeg', NULL, 'dandelion'),
('https://images.pexels.com/photos/413195/pexels-photo-413195.jpeg', NULL, 'grass'),
('https://images.pexels.com/photos/36762/grass-lawn-garden-green.jpg', NULL, 'grass');