-- =============================================================================
-- BS-Opt Database Schema
-- TimescaleDB + PostgreSQL Native Auth with OAuth Support
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;      -- Native password hashing
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";   -- UUID generation

-- =============================================================================
-- AUTHENTICATION (Native PostgreSQL with OAuth)
-- =============================================================================

-- OAuth providers enum
CREATE TYPE oauth_provider AS ENUM ('google', 'github', 'local');

-- Users table with OAuth and email verification
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT,  -- NULL for OAuth-only users
    role VARCHAR(50) NOT NULL DEFAULT 'trader' CHECK (role IN ('trader', 'researcher', 'admin')),
    is_active BOOLEAN DEFAULT true,
    
    -- Email verification
    email_verified BOOLEAN DEFAULT false,
    email_verification_token TEXT,
    email_verification_expires TIMESTAMPTZ,
    
    -- OAuth fields
    oauth_provider oauth_provider DEFAULT 'local',
    oauth_provider_id VARCHAR(255),  -- External ID from Google/GitHub
    oauth_access_token TEXT,
    oauth_refresh_token TEXT,
    
    -- Profile (populated from OAuth or manual)
    display_name VARCHAR(255),
    avatar_url TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT valid_auth CHECK (
        password_hash IS NOT NULL OR oauth_provider != 'local'
    ),
    CONSTRAINT unique_oauth UNIQUE (oauth_provider, oauth_provider_id)
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_oauth ON users(oauth_provider, oauth_provider_id);

-- =============================================================================
-- PASSWORD FUNCTIONS
-- =============================================================================

CREATE OR REPLACE FUNCTION hash_password(password TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN crypt(password, gen_salt('bf', 12));
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION verify_password(password TEXT, password_hash TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN password_hash = crypt(password, password_hash);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================================================
-- EMAIL VERIFICATION
-- =============================================================================

CREATE OR REPLACE FUNCTION generate_verification_token(user_email TEXT)
RETURNS TEXT AS $$
DECLARE
    token TEXT;
BEGIN
    token := encode(gen_random_bytes(32), 'hex');
    
    UPDATE users 
    SET 
        email_verification_token = crypt(token, gen_salt('bf', 8)),
        email_verification_expires = NOW() + INTERVAL '24 hours'
    WHERE email = user_email;
    
    RETURN token;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION verify_email(user_email TEXT, token TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    success BOOLEAN;
BEGIN
    UPDATE users 
    SET 
        email_verified = true,
        email_verification_token = NULL,
        email_verification_expires = NULL
    WHERE email = user_email
      AND email_verification_token = crypt(token, email_verification_token)
      AND email_verification_expires > NOW()
    RETURNING true INTO success;
    
    RETURN COALESCE(success, false);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================================================
-- LOCAL AUTHENTICATION
-- =============================================================================

CREATE OR REPLACE FUNCTION register_local_user(
    p_email TEXT,
    p_password TEXT,
    p_role TEXT DEFAULT 'trader'
)
RETURNS TABLE (user_id UUID, verification_token TEXT) AS $$
DECLARE
    new_user_id UUID;
    token TEXT;
BEGIN
    INSERT INTO users (email, password_hash, role, oauth_provider)
    VALUES (p_email, hash_password(p_password), p_role, 'local')
    RETURNING id INTO new_user_id;
    
    SELECT generate_verification_token(p_email) INTO token;
    
    RETURN QUERY SELECT new_user_id, token;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION authenticate_user(user_email TEXT, user_password TEXT)
RETURNS TABLE (user_id UUID, user_role VARCHAR(50), is_verified BOOLEAN) AS $$
BEGIN
    RETURN QUERY
    UPDATE users 
    SET last_login = NOW()
    WHERE email = user_email 
      AND is_active = true
      AND oauth_provider = 'local'
      AND verify_password(user_password, password_hash)
    RETURNING id, role, email_verified;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================================================
-- OAUTH AUTHENTICATION (Google/GitHub)
-- =============================================================================

CREATE OR REPLACE FUNCTION upsert_oauth_user(
    p_provider oauth_provider,
    p_provider_id TEXT,
    p_email TEXT,
    p_display_name TEXT DEFAULT NULL,
    p_avatar_url TEXT DEFAULT NULL,
    p_access_token TEXT DEFAULT NULL,
    p_refresh_token TEXT DEFAULT NULL
)
RETURNS TABLE (user_id UUID, user_role VARCHAR(50), is_new BOOLEAN) AS $$
DECLARE
    existing_user_id UUID;
    result_is_new BOOLEAN := false;
BEGIN
    -- Check if OAuth user exists
    SELECT id INTO existing_user_id
    FROM users
    WHERE oauth_provider = p_provider AND oauth_provider_id = p_provider_id;
    
    IF existing_user_id IS NULL THEN
        -- Check if email already exists (link accounts)
        SELECT id INTO existing_user_id
        FROM users WHERE email = p_email;
        
        IF existing_user_id IS NOT NULL THEN
            -- Link OAuth to existing account
            UPDATE users SET
                oauth_provider = p_provider,
                oauth_provider_id = p_provider_id,
                oauth_access_token = p_access_token,
                oauth_refresh_token = p_refresh_token,
                display_name = COALESCE(display_name, p_display_name),
                avatar_url = COALESCE(avatar_url, p_avatar_url),
                email_verified = true,  -- OAuth emails are pre-verified
                last_login = NOW()
            WHERE id = existing_user_id;
        ELSE
            -- Create new OAuth user
            INSERT INTO users (
                email, oauth_provider, oauth_provider_id,
                oauth_access_token, oauth_refresh_token,
                display_name, avatar_url, email_verified
            )
            VALUES (
                p_email, p_provider, p_provider_id,
                p_access_token, p_refresh_token,
                p_display_name, p_avatar_url, true
            )
            RETURNING id INTO existing_user_id;
            
            result_is_new := true;
        END IF;
    ELSE
        -- Update existing OAuth user
        UPDATE users SET
            oauth_access_token = p_access_token,
            oauth_refresh_token = COALESCE(p_refresh_token, oauth_refresh_token),
            display_name = COALESCE(p_display_name, display_name),
            avatar_url = COALESCE(p_avatar_url, avatar_url),
            last_login = NOW()
        WHERE id = existing_user_id;
    END IF;
    
    RETURN QUERY
    SELECT existing_user_id, role, result_is_new
    FROM users WHERE id = existing_user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================================================
-- SESSION MANAGEMENT
-- =============================================================================

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    device_info JSONB
);

CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_expires ON sessions(expires_at);

CREATE OR REPLACE FUNCTION create_session(
    p_user_id UUID,
    p_token TEXT,
    p_expires_hours INTEGER DEFAULT 24,
    p_ip INET DEFAULT NULL,
    p_user_agent TEXT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    session_id UUID;
BEGIN
    -- Clean up expired sessions for this user
    DELETE FROM sessions 
    WHERE user_id = p_user_id AND expires_at < NOW();
    
    INSERT INTO sessions (user_id, token_hash, expires_at, ip_address, user_agent)
    VALUES (
        p_user_id,
        crypt(p_token, gen_salt('bf', 8)),
        NOW() + (p_expires_hours || ' hours')::INTERVAL,
        p_ip,
        p_user_agent
    )
    RETURNING id INTO session_id;
    
    RETURN session_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION validate_session(p_token TEXT)
RETURNS TABLE (
    user_id UUID,
    user_role VARCHAR(50),
    user_email VARCHAR(255),
    email_verified BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT u.id, u.role, u.email, u.email_verified
    FROM sessions s
    JOIN users u ON s.user_id = u.id
    WHERE s.token_hash = crypt(p_token, s.token_hash)
      AND s.expires_at > NOW()
      AND u.is_active = true;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION revoke_session(p_token TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    deleted BOOLEAN;
BEGIN
    DELETE FROM sessions 
    WHERE token_hash = crypt(p_token, token_hash)
    RETURNING true INTO deleted;
    
    RETURN COALESCE(deleted, false);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================================================
-- MARKET DATA (TimescaleDB Hypertables)
-- =============================================================================

CREATE TABLE IF NOT EXISTS market_ticks (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8),
    bid DECIMAL(20, 8),
    ask DECIMAL(20, 8),
    source VARCHAR(50)
);

SELECT create_hypertable('market_ticks', 'time', if_not_exists => TRUE);
SELECT add_compression_policy('market_ticks', INTERVAL '1 day', if_not_exists => TRUE);

CREATE INDEX idx_ticks_symbol_time ON market_ticks (symbol, time DESC);

-- Continuous aggregate for OHLCV
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    symbol,
    exchange,
    first(price, time) AS open,
    max(price) AS high,
    min(price) AS low,
    last(price, time) AS close,
    sum(volume) AS volume
FROM market_ticks
GROUP BY bucket, symbol, exchange
WITH NO DATA;

SELECT add_continuous_aggregate_policy('ohlcv_1m',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- =============================================================================
-- PRICING & GREEKS
-- =============================================================================

CREATE TABLE IF NOT EXISTS option_greeks (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    underlying VARCHAR(20) NOT NULL,
    strike DECIMAL(20, 4) NOT NULL,
    expiry DATE NOT NULL,
    option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('call', 'put')),
    delta DECIMAL(10, 6),
    gamma DECIMAL(10, 6),
    theta DECIMAL(10, 6),
    vega DECIMAL(10, 6),
    rho DECIMAL(10, 6),
    theoretical_price DECIMAL(20, 8),
    market_price DECIMAL(20, 8),
    implied_vol DECIMAL(10, 6),
    model VARCHAR(50),
    computation_time_us INTEGER
);

SELECT create_hypertable('option_greeks', 'time', if_not_exists => TRUE);
CREATE INDEX idx_greeks_symbol ON option_greeks (symbol, time DESC);

-- =============================================================================
-- NUMERICAL EXPERIMENTS (Academic Research)
-- =============================================================================

CREATE TABLE IF NOT EXISTS numerical_experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    researcher_id UUID REFERENCES users(id),
    experiment_name VARCHAR(255),
    underlying_symbol VARCHAR(20),
    spot_price DECIMAL(20, 8),
    strike DECIMAL(20, 8),
    risk_free_rate DECIMAL(10, 6),
    volatility DECIMAL(10, 6),
    time_to_maturity DECIMAL(10, 6),
    option_type VARCHAR(4),
    analytical_price DECIMAL(20, 8),
    fdm_price DECIMAL(20, 8),
    fdm_time_us INTEGER,
    fdm_grid_size INTEGER,
    mc_price DECIMAL(20, 8),
    mc_time_us INTEGER,
    mc_paths INTEGER,
    mc_std_error DECIMAL(20, 8),
    tree_price DECIMAL(20, 8),
    tree_time_us INTEGER,
    tree_steps INTEGER,
    fdm_error_pct DECIMAL(10, 6),
    mc_error_pct DECIMAL(10, 6),
    tree_error_pct DECIMAL(10, 6),
    notes TEXT
);

CREATE INDEX idx_experiments_researcher ON numerical_experiments (researcher_id, created_at DESC);

-- =============================================================================
-- AUDIT LOGS (Immutable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS audit_logs (
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actor_id UUID,
    actor_email VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    request_id UUID,
    latency_ms INTEGER,
    status_code INTEGER,
    details JSONB
);

SELECT create_hypertable('audit_logs', 'time', if_not_exists => TRUE);
SELECT add_compression_policy('audit_logs', INTERVAL '1 hour', if_not_exists => TRUE);

CREATE INDEX idx_audit_actor ON audit_logs (actor_id, time DESC);
CREATE INDEX idx_audit_action ON audit_logs (action, time DESC);

-- =============================================================================
-- ROW LEVEL SECURITY
-- =============================================================================

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE numerical_experiments ENABLE ROW LEVEL SECURITY;

CREATE POLICY users_self_access ON users
    FOR ALL USING (
        id = current_setting('app.current_user_id', TRUE)::UUID
        OR current_setting('app.current_user_role', TRUE) = 'admin'
    );

CREATE POLICY experiments_researcher_access ON numerical_experiments
    FOR ALL USING (
        researcher_id = current_setting('app.current_user_id', TRUE)::UUID
        OR current_setting('app.current_user_role', TRUE) IN ('admin', 'researcher')
    );

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

CREATE OR REPLACE FUNCTION set_session_context(p_user_id UUID, p_role VARCHAR)
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.current_user_id', p_user_id::TEXT, FALSE);
    PERFORM set_config('app.current_user_role', p_role, FALSE);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Create MLflow database
CREATE DATABASE mlflow;
