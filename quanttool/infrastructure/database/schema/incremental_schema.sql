-- IncrementalDataManager Schema for PostgreSQL
-- Manages data ranges and update logs for incremental data fetching

-- Data ranges table (replaces data_ranges)
-- Tracks the available data range for each symbol and data type
CREATE TABLE IF NOT EXISTS data_ranges (
    symbol VARCHAR(20) NOT NULL,
    data_type VARCHAR(30) NOT NULL DEFAULT 'stock_bar',
    earliest_date DATE NOT NULL,
    latest_date DATE NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    file_path VARCHAR(500) NOT NULL,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    size_bytes BIGINT DEFAULT 0,
    PRIMARY KEY (symbol, data_type)
);

-- Update logs table (replaces update_log)
-- Records all data updates for audit and potential rollback
CREATE TABLE IF NOT EXISTS update_logs (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    data_type VARCHAR(30) NOT NULL DEFAULT 'stock_bar',
    update_type VARCHAR(30) NOT NULL,
    old_range_start DATE,
    old_range_end DATE,
    new_range_start DATE,
    new_range_end DATE,
    rows_added INTEGER DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_data_ranges_expires_at ON data_ranges(expires_at);
CREATE INDEX IF NOT EXISTS idx_data_ranges_data_type ON data_ranges(data_type);
CREATE INDEX IF NOT EXISTS idx_data_ranges_latest_date ON data_ranges(latest_date DESC);

CREATE INDEX IF NOT EXISTS idx_update_logs_symbol ON update_logs(symbol);
CREATE INDEX IF NOT EXISTS idx_update_logs_data_type ON update_logs(data_type);
CREATE INDEX IF NOT EXISTS idx_update_logs_timestamp ON update_logs(timestamp DESC);

-- Function to clean up expired data
CREATE OR REPLACE FUNCTION cleanup_expired_data()
RETURNS TABLE(symbol VARCHAR(20), data_type VARCHAR(30), file_path VARCHAR(500)) AS $$
BEGIN
    RETURN QUERY
    DELETE FROM data_ranges
    WHERE expires_at < NOW()
    RETURNING data_ranges.symbol, data_ranges.data_type, data_ranges.file_path;
END;
$$ LANGUAGE plpgsql;

-- Function to get data range with row-level lock
-- Used for concurrent updates to prevent race conditions
CREATE OR REPLACE FUNCTION get_or_lock_data_range(
    p_symbol VARCHAR(20),
    p_data_type VARCHAR(30) DEFAULT 'stock_bar'
)
RETURNS TABLE(
    symbol VARCHAR(20),
    data_type VARCHAR(30),
    earliest_date DATE,
    latest_date DATE,
    row_count INTEGER,
    file_path VARCHAR(500),
    last_updated TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        dr.symbol,
        dr.data_type,
        dr.earliest_date,
        dr.latest_date,
        dr.row_count,
        dr.file_path,
        dr.last_updated
    FROM data_ranges dr
    WHERE dr.symbol = p_symbol AND dr.data_type = p_data_type
    FOR UPDATE;
END;
$$ LANGUAGE plpgsql;

-- Function to record update log
CREATE OR REPLACE FUNCTION record_update_log(
    p_symbol VARCHAR(20),
    p_data_type VARCHAR(30),
    p_update_type VARCHAR(30),
    p_old_start DATE DEFAULT NULL,
    p_old_end DATE DEFAULT NULL,
    p_new_start DATE DEFAULT NULL,
    p_new_end DATE DEFAULT NULL,
    p_rows_added INTEGER DEFAULT 0
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO update_logs (
        symbol, data_type, update_type,
        old_range_start, old_range_end,
        new_range_start, new_range_end,
        rows_added
    ) VALUES (
        p_symbol, p_data_type, p_update_type,
        p_old_start, p_old_end,
        p_new_start, p_new_end,
        p_rows_added
    );
END;
$$ LANGUAGE plpgsql;

-- View for data coverage statistics
CREATE OR REPLACE VIEW data_coverage_stats AS
SELECT
    data_type,
    COUNT(*) as symbol_count,
    SUM(row_count) as total_rows,
    ROUND(SUM(size_bytes)::NUMERIC / 1024 / 1024, 2) as total_size_mb,
    MIN(earliest_date) as global_earliest,
    MAX(latest_date) as global_latest,
    AVG(latest_date - earliest_date) as avg_coverage_days
FROM data_ranges
GROUP BY data_type;

-- View for symbols needing update
CREATE OR REPLACE VIEW symbols_needing_update AS
SELECT
    symbol,
    data_type,
    latest_date,
    last_updated,
    expires_at,
    EXTRACT(DAY FROM (NOW() - last_updated)) as days_since_update
FROM data_ranges
WHERE expires_at < NOW()
ORDER BY expires_at ASC;
