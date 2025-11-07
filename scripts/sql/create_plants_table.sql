CREATE TABLE IF NOT EXISTS plants_data (
  id SERIAL PRIMARY KEY,
  url_source TEXT NOT NULL,
  url_s3 TEXT,
  label VARCHAR(32) NOT NULL CHECK (label IN ('dandelion','grass')),
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_plants_data_url_source ON plants_data ((md5(url_source)));

CREATE INDEX IF NOT EXISTS ix_plants_data_label ON plants_data (label);
CREATE INDEX IF NOT EXISTS ix_plants_data_url_s3_null ON plants_data ((url_s3 IS NULL));