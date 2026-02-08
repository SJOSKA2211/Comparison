export default function PositionsTable({ portfolio, onUpdate }) {
    const positions = portfolio?.positions || []
    const totalPnl = positions.reduce((sum, pos) => sum + (pos.current_price - pos.average_price) * pos.quantity * 100, 0)

    return (
        <div className="flex flex-col gap-lg">
            <div className="flex justify-between items-center">
                <div>
                    <h2>Open Positions</h2>
                    <p className="text-muted" style={{ fontSize: '0.8125rem', marginTop: '4px' }}>
                        {positions.length} active contracts
                    </p>
                </div>
                <div className="flex items-center gap-md">
                    <div className="stat">
                        <span className="stat__label">Total P&L</span>
                        <span className={`stat__value ${totalPnl >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                            {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
                        </span>
                    </div>
                </div>
            </div>

            <div className="card">
                <div className="card__body" style={{ padding: 0 }}>
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Type</th>
                                <th>Strike</th>
                                <th>Expiry</th>
                                <th>Qty</th>
                                <th>Avg Price</th>
                                <th>Current</th>
                                <th>P&L ($)</th>
                                <th>P&L (%)</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {positions.map(pos => (
                                <tr key={pos.id}>
                                    <td style={{ fontWeight: 600 }}>{pos.symbol}</td>
                                    <td>
                                        <span style={{
                                            display: 'inline-block',
                                            padding: '2px 6px',
                                            borderRadius: '4px',
                                            fontSize: '0.625rem',
                                            fontWeight: 600,
                                            background: pos.type === 'CALL' ? 'rgba(0, 230, 118, 0.15)' : 'rgba(255, 82, 82, 0.15)',
                                            color: pos.type === 'CALL' ? 'var(--bullish)' : 'var(--bearish)'
                                        }}>
                                            {pos.type}
                                        </span>
                                    </td>
                                    <td>${pos.strike}</td>
                                    <td>{pos.expiry}</td>
                                    <td style={{ color: pos.quantity < 0 ? 'var(--bearish)' : 'var(--pearl)' }}>
                                        {pos.quantity}
                                    </td>
                                    <td className="mono">${pos.average_price.toFixed(2)}</td>
                                    <td className="mono">${pos.current_price.toFixed(2)}</td>
                                    <td className={`mono ${pos.current_price >= pos.average_price ? 'text-bullish' : 'text-bearish'}`}>
                                        {((pos.current_price - pos.average_price) * pos.quantity * 100) >= 0 ? '+' : ''}{((pos.current_price - pos.average_price) * pos.quantity * 100).toFixed(2)}
                                    </td>
                                    <td className={`mono ${pos.current_price >= pos.average_price ? 'text-bullish' : 'text-bearish'}`}>
                                        {(((pos.current_price - pos.average_price) / pos.average_price) * 100) >= 0 ? '+' : ''}{(((pos.current_price - pos.average_price) / pos.average_price) * 100).toFixed(1)}%
                                    </td>
                                    <td>
                                        <div className="flex gap-xs">
                                            <button className="btn btn--secondary btn--sm">Close</button>
                                            <button className="btn btn--ghost btn--sm">Roll</button>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Greeks Summary */}
            <div className="card">
                <div className="card__header">
                    <span className="card__title">Portfolio Greeks</span>
                </div>
                <div className="card__body">
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 'var(--space-md)' }}>
                        <div className="stat">
                            <span className="stat__label">Delta</span>
                            <span className="stat__value stat__value--sm text-bullish">+342</span>
                        </div>
                        <div className="stat">
                            <span className="stat__label">Gamma</span>
                            <span className="stat__value stat__value--sm">+28.5</span>
                        </div>
                        <div className="stat">
                            <span className="stat__label">Theta</span>
                            <span className="stat__value stat__value--sm text-bearish">-156</span>
                        </div>
                        <div className="stat">
                            <span className="stat__label">Vega</span>
                            <span className="stat__value stat__value--sm">+84.2</span>
                        </div>
                        <div className="stat">
                            <span className="stat__label">Rho</span>
                            <span className="stat__value stat__value--sm">+12.8</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
