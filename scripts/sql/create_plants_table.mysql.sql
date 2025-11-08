-- scripts/sql/create_plants_data_mysql.sql
-- Purpose:
--   Recreate the "plants_data" table for MySQL.
--   This table is the *source of truth* for ETL: rows with url_s3 = NULL
--   still need to be downloaded and pushed to S3 by the pipeline.
--
-- How it is used in the project:
--   1) Airflow / CLI scripts read rows with url_s3 IS NULL
--   2) download image from url_source
--   3) upload to S3 (MinIO)
--   4) update url_s3 with "s3://bucket/key"

DROP TABLE IF EXISTS plants_data;

CREATE TABLE IF NOT EXISTS plants_data (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  url_source TEXT NOT NULL,            -- original URL (internet)
  url_s3     TEXT NULL,                -- destination URL after upload to MinIO
  label ENUM('dandelion','grass') NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),

  -- simple lookup index (fast SELECT WHERE label='grass' ...)
  KEY idx_label (label)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;