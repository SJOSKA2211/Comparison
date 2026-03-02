import { useState, useCallback } from 'react'

const defaultParams = {
    spot: 100,
    strike: 100,
    rate: 0.05,
    volatility: 0.20,
    timeToMaturity: 1.0,
    optionType: 'call'
}

export default function PricingPanel() {
    const [params, setParams] = useState(defaultParams)
    const [result, setResult] = useState(null)
    const [comparison, setComparison] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    const handleChange = (field, value) => {
        setParams(prev => ({ ...prev, [field]: value }))
    }

    const calculatePrice = useCallback(async () => {
        setLoading(true)
        setError(null)

        try {
            const token = localStorage.getItem('token')
            const res = await fetch('/api/pricing/black-scholes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(token && { 'Authorization': `Bearer ${token}` })
                },
                body: JSON.stringify({
                    spot: params.spot,
                    strike: params.strike,
                    rate: params.rate,
                    volatility: params.volatility,
                    time_to_maturity: params.timeToMaturity,
                    option_type: params.optionType
                })
            })

            if (!res.ok) throw new Error('Pricing failed')
            const data = await res.json()
            setResult(data)
        } catch (e) {
            setError(e.message)
        } finally {
            setLoading(false)
        }
    }, [params])

    const compareAll = useCallback(async () => {
        setLoading(true)
        try {
            const token = localStorage.getItem('token')
            const res = await fetch('/api/pricing/compare', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(token && { 'Authorization': `Bearer ${token}` })
                },
                body: JSON.stringify({
                    spot: params.spot,
                    strike: params.strike,
                    rate: params.rate,
                    volatility: params.volatility,
                    time_to_maturity: params.timeToMaturity,
                    option_type: params.optionType
                })
            })

            if (res.ok) {
                setComparison(await res.json())
            }
        } catch (e) {
            console.error(e)
        } finally {
            setLoading(false)
        }
    }, [params])

    return (
        <div className="flex flex-col gap-lg">
            {/* Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h2>Options Pricing Engine</h2>
                    <p className="text-muted" style={{ fontSize: '0.8125rem', marginTop: '4px' }}>
                        Black-Scholes analytical pricing with full Greeks
                    </p>
                </div>
                <div className="flex gap-sm">
                    <button className="btn btn--secondary btn--sm" onClick={calculatePrice} disabled={loading}>
                        {loading ? '...' : 'Calculate'}
                    </button>
                    <button className="btn btn--ghost btn--sm" onClick={compareAll} disabled={loading}>
                        Compare Methods
                    </button>
                </div>
            </div>

            {/* Parameters Grid */}
            <div className="card">
                <div className="card__header">
                    <span className="card__title">Parameters</span>
                    <span className="status status--live">Real-time</span>
                </div>
                <div className="card__body">
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-md)' }}>
                        <div className="input-group">
                            <label>Spot Price (S)</label>
                            <input
                                type="number"
                                className="input"
                                value={params.spot}
                                onChange={e => handleChange('spot', parseFloat(e.target.value))}
                            />
                        </div>
                        <div className="input-group">
                            <label>Strike Price (K)</label>
                            <input
                                type="number"
                                className="input"
                                value={params.strike}
                                onChange={e => handleChange('strike', parseFloat(e.target.value))}
                            />
                        </div>
                        <div className="input-group">
                            <label>Risk-Free Rate (r)</label>
                            <input
                                type="number"
                                className="input"
                                step="0.01"
                                value={params.rate}
                                onChange={e => handleChange('rate', parseFloat(e.target.value))}
                            />
                        </div>
                        <div className="input-group">
                            <label>Volatility (σ)</label>
                            <input
                                type="number"
                                className="input"
                                step="0.01"
                                value={params.volatility}
                                onChange={e => handleChange('volatility', parseFloat(e.target.value))}
                            />
                        </div>
                        <div className="input-group">
                            <label>Time to Maturity (T)</label>
                            <input
                                type="number"
                                className="input"
                                step="0.1"
                                value={params.timeToMaturity}
                                onChange={e => handleChange('timeToMaturity', parseFloat(e.target.value))}
                            />
                        </div>
                        <div className="input-group">
                            <label>Option Type</label>
                            <select
                                className="input"
                                value={params.optionType}
                                onChange={e => handleChange('optionType', e.target.value)}
                            >
                                <option value="call">Call</option>
                                <option value="put">Put</option>
                            </select>
                        </div>
                    </div>
                </div>
            </div>

            {/* Results */}
            {result && (
                <div className="card card--glass">
                    <div className="card__header">
                        <span className="card__title">Pricing Result</span>
                        {result.local && <span style={{ fontSize: '0.625rem', color: 'var(--neutral)' }}>Local calc</span>}
                    </div>
                    <div className="card__body">
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-lg)' }}>
                            <div className="stat">
                                <span className="stat__label">Option Price</span>
                                <span className="stat__value text-accent">${result.price?.toFixed(4)}</span>
                            </div>
                            <div className="stat">
                                <span className="stat__label">Delta (Δ)</span>
                                <span className="stat__value stat__value--sm">{result.delta?.toFixed(4)}</span>
                            </div>
                            <div className="stat">
                                <span className="stat__label">Gamma (Γ)</span>
                                <span className="stat__value stat__value--sm">{result.gamma?.toFixed(4)}</span>
                            </div>
                            <div className="stat">
                                <span className="stat__label">Theta (Θ)</span>
                                <span className="stat__value stat__value--sm">{result.theta?.toFixed(4)}</span>
                            </div>
                            <div className="stat">
                                <span className="stat__label">Vega (ν)</span>
                                <span className="stat__value stat__value--sm">{result.vega?.toFixed(4)}</span>
                            </div>
                            <div className="stat">
                                <span className="stat__label">Rho (ρ)</span>
                                <span className="stat__value stat__value--sm">{result.rho?.toFixed(4)}</span>
                            </div>
                            <div className="stat">
                                <span className="stat__label">Compute Time</span>
                                <span className="stat__value stat__value--sm">{result.computation_time_us || '—'} μs</span>
                            </div>
                            <div className="stat">
                                <span className="stat__label">Model</span>
                                <span className="stat__value stat__value--sm" style={{ textTransform: 'uppercase' }}>
                                    {result.model || 'BS'}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Method Comparison */}
            {comparison && (
                <div className="card">
                    <div className="card__header">
                        <span className="card__title">Method Comparison</span>
                    </div>
                    <div className="card__body" style={{ padding: 0 }}>
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Method</th>
                                    <th>Price</th>
                                    <th>Error %</th>
                                    <th>Time (μs)</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Analytical (BS)</td>
                                    <td>{comparison.analytical?.price?.toFixed(6)}</td>
                                    <td>—</td>
                                    <td>~1</td>
                                </tr>
                                <tr>
                                    <td>Crank-Nicolson FDM</td>
                                    <td>{comparison.fdm?.price?.toFixed(6)}</td>
                                    <td className={comparison.fdm?.error_pct < 0.1 ? 'text-bullish' : ''}>
                                        {comparison.fdm?.error_pct?.toFixed(4)}
                                    </td>
                                    <td>{comparison.fdm?.time_us}</td>
                                </tr>
                                <tr>
                                    <td>Monte Carlo</td>
                                    <td>{comparison.monte_carlo?.price?.toFixed(6)}</td>
                                    <td>{comparison.monte_carlo?.error_pct?.toFixed(4)}</td>
                                    <td>{comparison.monte_carlo?.time_us}</td>
                                </tr>
                                <tr>
                                    <td>Trinomial Tree</td>
                                    <td>{comparison.trinomial?.price?.toFixed(6)}</td>
                                    <td className={comparison.trinomial?.error_pct < 0.1 ? 'text-bullish' : ''}>
                                        {comparison.trinomial?.error_pct?.toFixed(4)}
                                    </td>
                                    <td>{comparison.trinomial?.time_us}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {error && (
                <div style={{ color: 'var(--bearish)', fontSize: '0.8125rem' }}>
                    Error: {error}
                </div>
            )}
        </div>
    )
}

