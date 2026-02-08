export default function Footer() {
    return (
        <footer className="terminal__footer">
            <div className="flex items-center gap-lg">
                <span className="status status--live">API Connected</span>
                <span>•</span>
                <span>Latency: <span className="mono text-accent">2.4ms</span></span>
                <span>•</span>
                <span>WebSocket: <span className="text-bullish">Active</span></span>
            </div>

            <div className="flex items-center gap-lg">
                <span>BS-Opt v1.0.0</span>
                <span>•</span>
                <span>© 2024 Quantitative Research</span>
            </div>
        </footer>
    )
}
