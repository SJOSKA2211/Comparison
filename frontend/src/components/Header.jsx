import { useState, useEffect } from 'react'

export default function Header({ user, onLogout }) {
    const [time, setTime] = useState(new Date())
    const [marketStatus, setMarketStatus] = useState('live')

    useEffect(() => {
        const timer = setInterval(() => setTime(new Date()), 1000)
        return () => clearInterval(timer)
    }, [])

    const formatTime = (date) => {
        return date.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        })
    }

    return (
        <header className="terminal__header">
            <div className="flex items-center gap-lg">
                <div className="flex items-center gap-sm">
                    <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                        <rect width="32" height="32" rx="8" fill="url(#logo-gradient)" />
                        <path d="M8 16L12 12L16 16L20 12L24 16" stroke="#0a0b0f" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                        <path d="M8 20L12 16L16 20L20 16L24 20" stroke="#0a0b0f" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                        <defs>
                            <linearGradient id="logo-gradient" x1="0" y1="0" x2="32" y2="32">
                                <stop stopColor="#00d9b8" />
                                <stop offset="1" stopColor="#00a88f" />
                            </linearGradient>
                        </defs>
                    </svg>
                    <div>
                        <div className="text-accent" style={{ fontSize: '0.875rem', fontWeight: 700, letterSpacing: '0.05em' }}>
                            BS-OPT
                        </div>
                        <div style={{ fontSize: '0.625rem', color: 'var(--silver)', letterSpacing: '0.1em' }}>
                            QUANT TERMINAL
                        </div>
                    </div>
                </div>

                <div className="status status--live">
                    Markets {marketStatus}
                </div>
            </div>

            <div className="flex items-center gap-lg">
                {/* Ticker preview */}
                <div className="flex gap-md" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                    <span>
                        <span style={{ color: 'var(--silver)' }}>AAPL</span>
                        <span className="text-bullish" style={{ marginLeft: '4px' }}>185.42</span>
                    </span>
                    <span>
                        <span style={{ color: 'var(--silver)' }}>SAFCOM</span>
                        <span className="text-bearish" style={{ marginLeft: '4px' }}>24.85</span>
                    </span>
                    <span>
                        <span style={{ color: 'var(--silver)' }}>BTC</span>
                        <span className="text-bullish" style={{ marginLeft: '4px' }}>67,432</span>
                    </span>
                </div>

                <div className="mono" style={{
                    fontSize: '1.25rem',
                    fontWeight: 600,
                    color: 'var(--white)',
                    letterSpacing: '0.02em'
                }}>
                    {formatTime(time)}
                </div>

                {user ? (
                    <div className="flex items-center gap-md">
                        <div style={{ fontSize: '0.75rem' }}>
                            <div style={{ color: 'var(--pearl)' }}>{user.email}</div>
                            <div style={{ color: 'var(--accent)', textTransform: 'uppercase', fontSize: '0.625rem' }}>
                                {user.role}
                            </div>
                        </div>
                        <button className="btn btn--ghost btn--sm" onClick={onLogout}>
                            Logout
                        </button>
                    </div>
                ) : (
                    <div style={{ fontSize: '0.75rem', color: 'var(--silver)' }}>
                        Not connected
                    </div>
                )}
            </div>
        </header>
    )
}
