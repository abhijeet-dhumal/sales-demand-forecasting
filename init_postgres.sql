-- ===================================================================
-- PostgreSQL Initialization Script for Feast Feature Store
-- ===================================================================
-- 
-- This script sets up the PostgreSQL databases and tables required
-- for the Feast feature store with the following architecture:
--
-- 1. feast_registry: Feature definitions & metadata
-- 2. feast_offline: Raw feature storage (sales, store data)
-- 3. feast_online: Low-latency serving features
--
-- PREREQUISITES:
-- - PostgreSQL 12+
-- - Run as postgres superuser or user with CREATEDB privileges
--
-- USAGE:
--   psql -U postgres -f init_postgres.sql
-- ===================================================================

-- Create databases
CREATE DATABASE feast_registry;
CREATE DATABASE feast_offline;
CREATE DATABASE feast_online;

\connect feast_registry

-- Create Feast user (if not exists)
DO
$$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'feast') THEN
      CREATE ROLE feast WITH LOGIN PASSWORD 'feast_password';
   END IF;
END
$$;

-- Grant permissions on feast_registry
GRANT ALL PRIVILEGES ON DATABASE feast_registry TO feast;
GRANT ALL PRIVILEGES ON SCHEMA public TO feast;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO feast;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO feast;

\connect feast_offline

-- Grant permissions on feast_offline
GRANT ALL PRIVILEGES ON DATABASE feast_offline TO feast;
GRANT ALL PRIVILEGES ON SCHEMA public TO feast;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO feast;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO feast;

-- ===================================================================
-- SALES FEATURES TABLE
-- ===================================================================
-- Historical sales data with pre-computed time-series features
-- Source: Walmart sales forecasting dataset (421K records)

CREATE TABLE IF NOT EXISTS sales_features (
    -- Entity keys
    store INTEGER NOT NULL,
    dept INTEGER NOT NULL,
    date TIMESTAMP NOT NULL,
    
    -- Target variable
    weekly_sales DOUBLE PRECISION NOT NULL,
    
    -- Holiday indicator
    is_holiday INTEGER NOT NULL,
    
    -- Temporal features (pre-computed in ETL)
    week_of_year INTEGER,
    month INTEGER,
    quarter INTEGER,
    
    -- Time-series features (pre-computed lags and rolling stats)
    sales_lag_1 DOUBLE PRECISION,
    sales_lag_2 DOUBLE PRECISION,
    sales_lag_4 DOUBLE PRECISION,
    sales_rolling_mean_4 DOUBLE PRECISION,
    sales_rolling_mean_12 DOUBLE PRECISION,
    sales_rolling_std_4 DOUBLE PRECISION,
    
    -- Composite primary key (entity + timestamp)
    PRIMARY KEY (store, dept, date)
);

-- Create indexes for fast retrieval
CREATE INDEX idx_sales_date ON sales_features(date);
CREATE INDEX idx_sales_store_dept ON sales_features(store, dept);
CREATE INDEX idx_sales_store_dept_date ON sales_features(store, dept, date);

COMMENT ON TABLE sales_features IS 'Historical sales with pre-computed time-series features';
COMMENT ON COLUMN sales_features.store IS 'Walmart store number (1-45)';
COMMENT ON COLUMN sales_features.dept IS 'Department number (1-99)';
COMMENT ON COLUMN sales_features.weekly_sales IS 'Weekly sales amount in USD (target variable)';
COMMENT ON COLUMN sales_features.sales_lag_1 IS 'Sales from 1 week ago (t-1)';
COMMENT ON COLUMN sales_features.sales_rolling_mean_4 IS '4-week rolling average of sales';

-- ===================================================================
-- STORE FEATURES TABLE
-- ===================================================================
-- Store-level external factors and metadata

CREATE TABLE IF NOT EXISTS store_features (
    -- Entity keys
    store INTEGER NOT NULL,
    dept INTEGER NOT NULL,
    date TIMESTAMP NOT NULL,
    
    -- Economic indicators
    temperature DOUBLE PRECISION,
    fuel_price DOUBLE PRECISION,
    cpi DOUBLE PRECISION,
    unemployment DOUBLE PRECISION,
    
    -- Markdown/promotional features
    markdown1 DOUBLE PRECISION,
    markdown2 DOUBLE PRECISION,
    markdown3 DOUBLE PRECISION,
    markdown4 DOUBLE PRECISION,
    markdown5 DOUBLE PRECISION,
    total_markdown DOUBLE PRECISION,
    has_markdown INTEGER,
    
    -- Store attributes (static, but denormalized for performance)
    store_type VARCHAR(10),
    store_size INTEGER,
    
    -- Composite primary key
    PRIMARY KEY (store, dept, date)
);

-- Create indexes for fast joins
CREATE INDEX idx_store_date ON store_features(date);
CREATE INDEX idx_store_store_dept ON store_features(store, dept);
CREATE INDEX idx_store_store_dept_date ON store_features(store, dept, date);

COMMENT ON TABLE store_features IS 'Store-level external factors and metadata';
COMMENT ON COLUMN store_features.temperature IS 'Regional temperature in Fahrenheit';
COMMENT ON COLUMN store_features.fuel_price IS 'Regional fuel price in $/gallon';
COMMENT ON COLUMN store_features.cpi IS 'Consumer Price Index';
COMMENT ON COLUMN store_features.unemployment IS 'Regional unemployment rate (%)';
COMMENT ON COLUMN store_features.total_markdown IS 'Total promotional markdown amount ($)';
COMMENT ON COLUMN store_features.store_type IS 'Store type: A (largest), B (medium), C (smallest)';
COMMENT ON COLUMN store_features.store_size IS 'Store square footage';

-- ===================================================================
-- FEAST ONLINE STORE SETUP
-- ===================================================================

\connect feast_online

-- Grant permissions on feast_online
GRANT ALL PRIVILEGES ON DATABASE feast_online TO feast;
GRANT ALL PRIVILEGES ON SCHEMA public TO feast;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO feast;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO feast;

-- Feast will automatically create online store tables during materialization
-- Tables typically follow the pattern: <feature_view_name>_<timestamp>
-- Example: sales_history_features_1709876543

-- ===================================================================
-- VERIFICATION & STATISTICS
-- ===================================================================

\connect feast_offline

-- Analyze tables for query optimization
ANALYZE sales_features;
ANALYZE store_features;

-- Display table statistics
SELECT 
    'feast_registry' as database,
    COUNT(*) as table_count
FROM pg_tables 
WHERE schemaname = 'public'
UNION ALL
SELECT 
    'feast_offline' as database,
    COUNT(*) as table_count
FROM pg_tables 
WHERE schemaname = 'public'
UNION ALL
SELECT 
    'feast_online' as database,
    COUNT(*) as table_count
FROM pg_tables 
WHERE schemaname = 'public';

-- Show created tables
\dt

-- Display index information
SELECT 
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

-- ===================================================================
-- SETUP COMPLETE
-- ===================================================================

\echo ''
\echo '================================================================='
\echo 'PostgreSQL Initialization Complete!'
\echo '================================================================='
\echo ''
\echo 'Created databases:'
\echo '  ✓ feast_registry - Feature definitions & metadata'
\echo '  ✓ feast_offline  - Raw feature storage (sales, store data)'
\echo '  ✓ feast_online   - Low-latency serving features'
\echo ''
\echo 'Created tables in feast_offline:'
\echo '  ✓ sales_features - Historical sales with time-series features'
\echo '  ✓ store_features - External factors & store metadata'
\echo ''
\echo 'Created user:'
\echo '  ✓ feast (password: feast_password)'
\echo ''
\echo 'Next steps:'
\echo '  1. Load data: Run notebook 01_data_preparation_feast_setup.ipynb'
\echo '  2. Apply Feast: feast -c feature_repo apply'
\echo '  3. Train model: Run notebook 02_distributed_training_kubeflow.ipynb'
\echo ''
\echo 'Connection strings:'
\echo '  Registry: postgresql+psycopg://feast:feast_password@localhost:5432/feast_registry'
\echo '  Offline:  postgresql+psycopg://feast:feast_password@localhost:5432/feast_offline'
\echo '  Online:   postgresql+psycopg://feast:feast_password@localhost:5432/feast_online'
\echo ''
\echo '================================================================='



