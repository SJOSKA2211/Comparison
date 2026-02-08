const positions = [
    { id: 1, symbol: 'AAPL', type: 'CALL', strike: 185, expiry: '2024-03-15', qty: 10, avgPrice: 8.45, currentPrice: 10.20, pnl: 175.00, pnlPct: 20.7 },
    { id: 2, symbol: 'GOOGL', type: 'PUT', strike: 145, expiry: '2024-03-22', qty: -5, avgPrice: 4.20, currentPrice: 3.80, pnl: 20.00, pnlPct: 9.5 },
    { id: 3, symbol: 'MSFT', type: 'CALL', strike: 410, expiry: '2024-04-19', qty: 15, avgPrice: 12.30, currentPrice: 11.85, pnl: -67.50, pnlPct: -3.7 },
    { id: 4, symbol: 'SAFCOM', type: 'CALL', strike: 25, expiry: '2024-06-21', qty: 50, avgPrice: 1.20, currentPrice: 1.45, pnl: 125.00, pnlPct: 20.8 },
    { id: 5, symbol: 'EQTY', type: 'PUT', strike: 45, expiry: '2024-05-17', qty: 20, avgPrice: 2.10, currentPrice: 2.05, pnl: -10.00, pnlPct: -2.4 },
]

export default function PositionsTable() {
    const totalPnl = positions.reduce((sum, p) => sum + p.pnl, 0)

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
                                <th>P&L</th>
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
                                    <td style={{ color: pos.qty < 0 ? 'var(--bearish)' : 'var(--pearl)' }}>
                                        {pos.qty}
                                    </td>
                                    <td>${pos.avgPrice.toFixed(2)}</td>
                                    <td>${pos.currentPrice.toFixed(2)}</td>
                                    <td>
                                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                                            <span className={pos.pnl >= 0 ? 'text-bullish' : 'text-bearish'}>
                                                {pos.pnl >= 0 ? '+' : ''}${pos.pnl.toFixed(2)}
                                            </span>
                                            <span style={{ fontSize: '0.625rem', color: pos.pnl >= 0 ? 'var(--bullish)' : 'var(--bearish)' }}>
                                                {pos.pnl >= 0 ? '+' : ''}{pos.pnlPct.toFixed(1)}%
                                            </span>
                                        </div>
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
