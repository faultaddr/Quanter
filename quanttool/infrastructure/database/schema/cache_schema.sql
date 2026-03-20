-- LocalDataCache Schema for PostgreSQL
-- Manages local cache metadata for stock data

-- Cache entries table (replaces cache_meta)
-- Tracks cached data with TTL-based expiration
CREATE TABLE IF NOT EXISTS cache_entries (
    cache_key VARCHAR(64) PRIMARY KEY,
    file_path VARCHAR(500) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    data_hash VARCHAR(64),
    row_count INTEGER DEFAULT 0,
    size_bytes BIGINT DEFAULT 0,
    symbol VARCHAR(20),
    start_date DATE,
    end_date DATE,
    timeframe VARCHAR(10) DEFAULT '1d'
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_cache_entries_expires_at ON cache_entries(expires_at);
CREATE INDEX IF NOT EXISTS idx_cache_entries_symbol ON cache_entries(symbol);
CREATE INDEX IF NOT EXISTS idx_cache_entries_created_at ON cache_entries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cache_entries_date_range ON cache_entries(symbol, start_date, end_date);

-- Function to clean up expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS TABLE(cache_key VARCHAR(64), file_path VARCHAR(500)) AS $$
BEGIN
    RETURN QUERY
    DELETE FROM cache_entries
    WHERE expires_at < NOW()
    RETURNING cache_entries.cache_key, cache_entries.file_path;
END;
$$ LANGUAGE plpgsql;

-- Function to get cache entry with lock
CREATE OR REPLACE FUNCTION get_cache_entry_locked(
    p_cache_key VARCHAR(64)
)
RETURNS TABLE(
    cache_key VARCHAR(64),
    file_path VARCHAR(500),
    created_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    data_hash VARCHAR(64),
    row_count INTEGER,
    size_bytes BIGINT,
    symbol VARCHAR(20),
    start_date DATE,
    end_date DATE,
    timeframe VARCHAR(10)
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ce.cache_key,
        ce.file_path,
        ce.created_at,
        ce.expires_at,
        ce.data_hash,
        ce.row_count,
        ce.size_bytes,
        ce.symbol,
        ce.start_date,
        ce.end_date,
        ce.timeframe
    FROM cache_entries ce
    WHERE ce.cache_key = p_cache_key
    FOR UPDATE;
END;
$$ LANGUAGE plpgsql;

-- Function to upsert cache entry
CREATE OR REPLACE FUNCTION upsert_cache_entry(
    p_cache_key VARCHAR(64),
    p_file_path VARCHAR(500),
    p_expires_at TIMESTAMPTZ,
    p_data_hash VARCHAR(64) DEFAULT NULL,
    p_row_count INTEGER DEFAULT 0,
    p_size_bytes BIGINT DEFAULT 0,
    p_symbol VARCHAR(20) DEFAULT NULL,
    p_start_date DATE DEFAULT NULL,
    p_end_date DATE DEFAULT NULL,
    p_timeframe VARCHAR(10) DEFAULT '1d'
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO cache_entries (
        cache_key, file_path, expires_at,
        data_hash, row_count, size_bytes,
        symbol, start_date, end_date, timeframe
    ) VALUES (
        p_cache_key, p_file_path, p_expires_at,
        p_data_hash, p_row_count, p_size_bytes,
        p_symbol, p_start_date, p_end_date, p_timeframe
    )
    ON CONFLICT (cache_key) DO UPDATE SET
        file_path = EXCLUDED.file_path,
        created_at = NOW(),
        expires_at = EXCLUDED.expires_at,
        data_hash = EXCLUDED.data_hash,
        row_count = EXCLUDED.row_count,
        size_bytes = EXCLUDED.size_bytes,
        symbol = EXCLUDED.symbol,
        start_date = EXCLUDED.start_date,
        end_date = EXCLUDED.end_date,
        timeframe = EXCLUDED.timeframe;
END;
$$ LANGUAGE plpgsql;

-- View for cache statistics
CREATE OR REPLACE VIEW cache_stats AS
SELECT
    COUNT(*) as entry_count,
    COALESCE(SUM(row_count), 0) as total_rows,
    COALESCE(SUM(size_bytes), 0) as total_size_bytes,
    ROUND(COALESCE(SUM(size_bytes), 0)::NUMERIC / 1024 / 1024, 2) as total_size_mb,
    COUNT(CASE WHEN expires_at > NOW() THEN 1 END) as active_entries,
    COUNT(CASE WHEN expires_at <= NOW() THEN 1 END) as expired_entries
FROM cache_entries;

-- View for cache by symbol
CREATE OR REPLACE VIEW cache_by_symbol AS
SELECT
    symbol,
    COUNT(*) as cache_count,
    SUM(row_count) as total_rows,
    ROUND(SUM(size_bytes)::NUMERIC / 1024 / 1024, 2) as total_size_mb,
    MIN(start_date) as earliest_date,
    MAX(end_date) as latest_date
FROM cache_entries
WHERE symbol IS NOT NULL
GROUP BY symbol
ORDER BY total_rows DESC;
