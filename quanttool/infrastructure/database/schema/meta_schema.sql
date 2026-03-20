-- MetaDB Schema for PostgreSQL
-- Converts SQLite TEXT fields to appropriate PostgreSQL types

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Experiment runs table
CREATE TABLE IF NOT EXISTS experiment_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type VARCHAR(100) NOT NULL,
    parameters JSONB DEFAULT '{}',
    git_commit VARCHAR(40),
    data_version VARCHAR(50),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'pending',
    results JSONB,
    artifacts JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    parameters JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    result JSONB,
    error TEXT
);

-- Symbols table (stock metadata)
CREATE TABLE IF NOT EXISTS symbols (
    symbol VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100),
    industry VARCHAR(50),
    market VARCHAR(20),
    list_date DATE,
    delist_date DATE,
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Scan records table
CREATE TABLE IF NOT EXISTS scan_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scan_type VARCHAR(50) NOT NULL,
    scan_date DATE NOT NULL,
    parameters JSONB DEFAULT '{}',
    total_count INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Scan stock results table
CREATE TABLE IF NOT EXISTS scan_stock_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scan_id UUID NOT NULL REFERENCES scan_records(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    score NUMERIC(10, 4),
    rank INTEGER,
    data JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(scan_id, symbol)
);

-- Portfolio backtests table
CREATE TABLE IF NOT EXISTS portfolio_backtests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    parameters JSONB DEFAULT '{}',
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital NUMERIC(15, 2) NOT NULL,
    final_capital NUMERIC(15, 2),
    total_return NUMERIC(10, 4),
    annualized_return NUMERIC(10, 4),
    max_drawdown NUMERIC(10, 4),
    sharpe_ratio NUMERIC(10, 4),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Portfolio holdings table
CREATE TABLE IF NOT EXISTS portfolio_holdings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    backtest_id UUID NOT NULL REFERENCES portfolio_backtests(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    quantity NUMERIC(15, 4) NOT NULL,
    entry_price NUMERIC(15, 4) NOT NULL,
    entry_date DATE NOT NULL,
    exit_price NUMERIC(15, 4),
    exit_date DATE,
    pnl NUMERIC(15, 4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Daily values table
CREATE TABLE IF NOT EXISTS daily_values (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    backtest_id UUID NOT NULL REFERENCES portfolio_backtests(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    portfolio_value NUMERIC(15, 2) NOT NULL,
    cash_value NUMERIC(15, 2) NOT NULL,
    positions_value NUMERIC(15, 2) NOT NULL,
    daily_return NUMERIC(10, 4),
    cumulative_return NUMERIC(10, 4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(backtest_id, date)
);

-- Email config table
CREATE TABLE IF NOT EXISTS email_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    smtp_host VARCHAR(100) NOT NULL,
    smtp_port INTEGER NOT NULL,
    smtp_user VARCHAR(100) NOT NULL,
    smtp_password VARCHAR(200) NOT NULL,
    from_address VARCHAR(100) NOT NULL,
    to_addresses JSONB DEFAULT '[]',
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_experiment_runs_type ON experiment_runs(type);
CREATE INDEX IF NOT EXISTS idx_experiment_runs_status ON experiment_runs(status);
CREATE INDEX IF NOT EXISTS idx_experiment_runs_created_at ON experiment_runs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_tasks_type ON tasks(type);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_symbols_industry ON symbols(industry);
CREATE INDEX IF NOT EXISTS idx_symbols_status ON symbols(status);

CREATE INDEX IF NOT EXISTS idx_scan_records_scan_type ON scan_records(scan_type);
CREATE INDEX IF NOT EXISTS idx_scan_records_scan_date ON scan_records(scan_date DESC);
CREATE INDEX IF NOT EXISTS idx_scan_records_status ON scan_records(status);

CREATE INDEX IF NOT EXISTS idx_scan_stock_results_scan_id ON scan_stock_results(scan_id);
CREATE INDEX IF NOT EXISTS idx_scan_stock_results_symbol ON scan_stock_results(symbol);
CREATE INDEX IF NOT EXISTS idx_scan_stock_results_score ON scan_stock_results(score DESC);

CREATE INDEX IF NOT EXISTS idx_portfolio_backtests_strategy ON portfolio_backtests(strategy);
CREATE INDEX IF NOT EXISTS idx_portfolio_backtests_status ON portfolio_backtests(status);
CREATE INDEX IF NOT EXISTS idx_portfolio_backtests_date_range ON portfolio_backtests(start_date, end_date);

CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_backtest_id ON portfolio_holdings(backtest_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_symbol ON portfolio_holdings(symbol);

CREATE INDEX IF NOT EXISTS idx_daily_values_backtest_id ON daily_values(backtest_id);
CREATE INDEX IF NOT EXISTS idx_daily_values_date ON daily_values(date);

-- Updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_experiment_runs_updated_at
    BEFORE UPDATE ON experiment_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_symbols_updated_at
    BEFORE UPDATE ON symbols
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_email_configs_updated_at
    BEFORE UPDATE ON email_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
