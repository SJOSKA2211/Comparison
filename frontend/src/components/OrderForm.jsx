import { useState } from 'react'

export default function OrderForm({ user, onLogin, portfolio, onOrderPlaced }) {
    const [mode, setMode] = useState(user ? 'order' : 'login')
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

    // Order state
    const [orderType, setOrderType] = useState('market')
    const [side, setSide] = useState('buy')
    const [quantity, setQuantity] = useState(100)
    const [limitPrice, setLimitPrice] = useState('')

    const handleLogin = async (e) => {
        e.preventDefault()
        setLoading(true)
        setError('')

        const result = await onLogin(email, password)
        if (!result.success) {
            setError(result.error)
        } else {
            setMode('order')
        }
        setLoading(false)
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        if (!portfolio) {
            setError('Please selecyt a portfolio')
            return
        }
        setLoading(true)
        setError('')

        try {
            const token = localStorage.getItem('token')
            const res = await fetch('/api/trading/orders', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    portfolio_id: portfolio.id,
                    symbol: 'AAPL', // Hardcoded for now, should come from active symbol
                    side: side.toUpperCase(),
                    order_type: orderType,
                    quantity: quantity,
                    price: orderType === 'market' ? null : parseFloat(limitPrice)
                })
            })

            if (!res.ok) {
                const errData = await res.json()
                throw new Error(errData.detail || 'Order failed')
            }

            if (onOrderPlaced) onOrderPlaced()
            alert('Order placed successfully!')
        } catch (e) {
            setError(e.message)
        } finally {
            setLoading(false)
        }
    }

    if (!user && mode !== 'order') {
        return (
            <div className="flex flex-col gap-md">
                <div style={{
                    fontSize: '0.625rem',
                    color: 'var(--silver)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.1em'
                }}>
                    Authentication
                </div>

                <div className="card">
                    <div className="card__header">
                        <span className="card__title">Login</span>
                    </div>
                    <div className="card__body">
                        <form onSubmit={handleLogin} className="flex flex-col gap-md">
                            <div className="input-group">
                                <label>Email</label>
                                <input
                                    type="email"
                                    className="input"
                                    value={email}
                                    onChange={e => setEmail(e.target.value)}
                                    placeholder="trader@bsopt.io"
                                    required
                                />
                            </div>
                            <div className="input-group">
                                <label>Password</label>
                                <input
                                    type="password"
                                    className="input"
                                    value={password}
                                    onChange={e => setPassword(e.target.value)}
                                    placeholder="••••••••"
                                    required
                                />
                            </div>

                            {error && (
                                <div style={{ color: 'var(--bearish)', fontSize: '0.75rem' }}>
                                    {error}
                                </div>
                            )}

                            <button type="submit" className="btn btn--primary w-full" disabled={loading}>
                                {loading ? 'Connecting...' : 'Connect'}
                            </button>
                        </form>

                        <div style={{
                            marginTop: 'var(--space-md)',
                            paddingTop: 'var(--space-md)',
                            borderTop: '1px solid var(--graphite)',
                            fontSize: '0.75rem',
                            color: 'var(--silver)',
                            textAlign: 'center'
                        }}>
                            <span>Or continue with</span>
                            <div className="flex gap-sm mt-md" style={{ justifyContent: 'center' }}>
                                <button className="btn btn--secondary btn--sm">Google</button>
                                <button className="btn btn--secondary btn--sm">GitHub</button>
                            </div>
                        </div>
                    </div>
                </div>

                <div style={{ fontSize: '0.6875rem', color: 'var(--silver)', textAlign: 'center' }}>
                    Demo: test@example.com / password
                </div>
            </div>
        )
    }

    return (
        <div className="flex flex-col gap-md">
            <div style={{
                fontSize: '0.625rem',
                color: 'var(--silver)',
                textTransform: 'uppercase',
                letterSpacing: '0.1em'
            }}>
                Order Entry
            </div>

            <div className="card">
                <div className="card__header">
                    <span className="card__title">New Order</span>
                    <span className="status status--live">Ready</span>
                </div>
                <div className="card__body">
                    {/* Side Toggle */}
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: '1fr 1fr',
                        gap: '2px',
                        marginBottom: 'var(--space-md)',
                        background: 'var(--graphite)',
                        borderRadius: 'var(--radius-md)',
                        padding: '2px'
                    }}>
                        <button
                            onClick={() => setSide('buy')}
                            style={{
                                padding: 'var(--space-sm)',
                                border: 'none',
                                borderRadius: 'var(--radius-sm)',
                                background: side === 'buy' ? 'var(--bullish)' : 'transparent',
                                color: side === 'buy' ? 'var(--void)' : 'var(--silver)',
                                fontWeight: 600,
                                cursor: 'pointer',
                                transition: 'all 150ms ease'
                            }}
                        >
                            BUY
                        </button>
                        <button
                            onClick={() => setSide('sell')}
                            style={{
                                padding: 'var(--space-sm)',
                                border: 'none',
                                borderRadius: 'var(--radius-sm)',
                                background: side === 'sell' ? 'var(--bearish)' : 'transparent',
                                color: side === 'sell' ? 'var(--white)' : 'var(--silver)',
                                fontWeight: 600,
                                cursor: 'pointer',
                                transition: 'all 150ms ease'
                            }}
                        >
                            SELL
                        </button>
                    </div>

                    {/* Order Type */}
                    <div className="input-group" style={{ marginBottom: 'var(--space-md)' }}>
                        <label>Order Type</label>
                        <select
                            className="input"
                            value={orderType}
                            onChange={e => setOrderType(e.target.value)}
                        >
                            <option value="market">Market</option>
                            <option value="limit">Limit</option>
                            <option value="stop">Stop</option>
                        </select>
                    </div>

                    {/* Quantity */}
                    <div className="input-group" style={{ marginBottom: 'var(--space-md)' }}>
                        <label>Contracts</label>
                        <input
                            type="number"
                            className="input"
                            value={quantity}
                            onChange={e => setQuantity(parseInt(e.target.value))}
                            min="1"
                        />
                    </div>

                    {/* Limit Price */}
                    {orderType !== 'market' && (
                        <div className="input-group" style={{ marginBottom: 'var(--space-md)' }}>
                            <label>{orderType === 'limit' ? 'Limit Price' : 'Stop Price'}</label>
                            <input
                                type="number"
                                step="0.01"
                                className="input"
                                value={limitPrice}
                                onChange={e => setLimitPrice(e.target.value)}
                                placeholder="0.00"
                            />
                        </div>
                    )}

                    {/* Summary */}
                    <div style={{
                        padding: 'var(--space-md)',
                        background: 'var(--slate)',
                        borderRadius: 'var(--radius-md)',
                        marginBottom: 'var(--space-md)'
                    }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 'var(--space-xs)' }}>
                            <span className="text-muted">Est. Premium</span>
                            <span className="mono">${(quantity * 10.55).toFixed(2)}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 'var(--space-xs)' }}>
                            <span className="text-muted">Commission</span>
                            <span className="mono">$0.65</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', fontWeight: 600, color: 'var(--white)' }}>
                            <span>Total</span>
                            <span className="mono">${(quantity * 10.55 + 0.65).toFixed(2)}</span>
                        </div>
                    </div>

                    {/* Submit */}
                    <button
                        onClick={handleSubmit}
                        disabled={loading || !portfolio}
                        className={`btn w-full ${side === 'buy' ? 'btn--primary' : 'btn--danger'}`}
                        style={{ padding: 'var(--space-md)' }}
                    >
                        {loading ? 'Processing...' : `${side.toUpperCase()} ${quantity} Contracts`}
                    </button>
                    {error && <div style={{ color: 'var(--bearish)', fontSize: '0.75rem', marginTop: 'var(--space-sm)' }}>{error}</div>}
                </div>
            </div>

            {/* Quick Stats */}
            <div className="card">
                <div className="card__body">
                    <div className="flex flex-col gap-sm">
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                            <span className="text-muted">Buying Power</span>
                            <span className="mono text-bullish">${(portfolio?.cash_balance || 0).toLocaleString()}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                            <span className="text-muted">Portfolio</span>
                            <span className="mono">{portfolio?.name || '—'}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                            <span className="text-muted">Open Positions</span>
                            <span className="mono">{portfolio?.positions?.length || 0}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
