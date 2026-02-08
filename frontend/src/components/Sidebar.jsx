const navItems = [
    { id: 'pricing', label: 'Options Pricing', icon: '◎' },
    { id: 'positions', label: 'Positions', icon: '▦' },
    { id: 'research', label: 'Research Lab', icon: '◇' },
    { id: 'signals', label: 'ML Signals', icon: '◈' },
    { id: 'settings', label: 'Settings', icon: '⚙' },
]

const markets = [
    { symbol: 'AAPL', name: 'Apple Inc', price: 185.42, change: 2.34, pctChange: 1.28 },
    { symbol: 'GOOGL', name: 'Alphabet', price: 141.80, change: -0.56, pctChange: -0.39 },
    { symbol: 'MSFT', name: 'Microsoft', price: 404.22, change: 5.18, pctChange: 1.30 },
    { symbol: 'SAFCOM', name: 'Safaricom PLC', price: 24.85, change: -0.35, pctChange: -1.39 },
    { symbol: 'EQTY', name: 'Equity Group', price: 44.50, change: 0.75, pctChange: 1.71 },
]

export default function Sidebar({ activeTab, onTabChange }) {
    return (
        <aside className="terminal__sidebar">
            {/* Navigation */}
            <nav style={{ marginBottom: 'var(--space-xl)' }}>
                <div style={{
                    fontSize: '0.625rem',
                    color: 'var(--silver)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.1em',
                    marginBottom: 'var(--space-sm)'
                }}>
                    Navigation
                </div>

                {navItems.map(item => (
                    <button
                        key={item.id}
                        onClick={() => onTabChange(item.id)}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 'var(--space-sm)',
                            width: '100%',
                            padding: 'var(--space-sm) var(--space-md)',
                            marginBottom: '2px',
                            background: activeTab === item.id ? 'var(--accent-glow)' : 'transparent',
                            border: 'none',
                            borderRadius: 'var(--radius-md)',
                            borderLeft: activeTab === item.id ? '3px solid var(--accent)' : '3px solid transparent',
                            color: activeTab === item.id ? 'var(--accent)' : 'var(--silver)',
                            fontSize: '0.8125rem',
                            fontFamily: 'var(--font-sans)',
                            cursor: 'pointer',
                            transition: 'all 150ms ease',
                            textAlign: 'left',
                        }}
                    >
                        <span style={{ fontSize: '1rem', opacity: 0.8 }}>{item.icon}</span>
                        {item.label}
                    </button>
                ))}
            </nav>

            {/* Watchlist */}
            <div>
                <div style={{
                    fontSize: '0.625rem',
                    color: 'var(--silver)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.1em',
                    marginBottom: 'var(--space-sm)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                }}>
                    <span>Watchlist</span>
                    <span style={{ color: 'var(--accent)', cursor: 'pointer' }}>+</span>
                </div>

                <div className="card">
                    {markets.map((m, i) => (
                        <div
                            key={m.symbol}
                            style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                padding: 'var(--space-sm) var(--space-md)',
                                borderBottom: i < markets.length - 1 ? '1px solid var(--graphite)' : 'none',
                                cursor: 'pointer',
                                transition: 'background 150ms ease',
                            }}
                            onMouseEnter={e => e.currentTarget.style.background = 'rgba(0, 217, 184, 0.05)'}
                            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                        >
                            <div>
                                <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, fontSize: '0.8125rem' }}>
                                    {m.symbol}
                                </div>
                                <div style={{ fontSize: '0.625rem', color: 'var(--silver)' }}>
                                    {m.name}
                                </div>
                            </div>
                            <div style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                                <div style={{ fontSize: '0.8125rem' }}>{m.price.toFixed(2)}</div>
                                <div style={{
                                    fontSize: '0.625rem',
                                    color: m.change >= 0 ? 'var(--bullish)' : 'var(--bearish)'
                                }}>
                                    {m.change >= 0 ? '▲' : '▼'} {Math.abs(m.pctChange).toFixed(2)}%
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </aside>
    )
}
