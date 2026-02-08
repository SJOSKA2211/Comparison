import { useState, useCallback } from 'react'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import PricingPanel from './components/PricingPanel'
import PositionsTable from './components/PositionsTable'
import OrderForm from './components/OrderForm'
import Footer from './components/Footer'

const API_URL = '/api'

export default function App() {
    const [user, setUser] = useState(null)
    const [portfolios, setPortfolios] = useState([])
    const [activePortfolio, setActivePortfolio] = useState(null)
    const [activeTab, setActiveTab] = useState('pricing')

    const fetchPortfolios = useCallback(async (token) => {
        try {
            const res = await fetch(`${API_URL}/trading/portfolios`, {
                headers: { 'Authorization': `Bearer ${token}` }
            })
            if (res.ok) {
                const data = await res.json()
                setPortfolios(data)
                if (data.length > 0) setActivePortfolio(data[0])
            }
        } catch (e) { console.error(e) }
    }, [])

    const handleLogin = useCallback(async (email, password) => {
        try {
            const res = await fetch(`${API_URL}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            })
            if (res.ok) {
                const data = await res.json()
                localStorage.setItem('token', data.access_token)
                setUser(data.user)
                await fetchPortfolios(data.access_token)
                return { success: true }
            }
            return { success: false, error: 'Invalid credentials' }
        } catch (e) {
            return { success: false, error: 'Connection failed' }
        }
    }, [])

    const handleLogout = useCallback(() => {
        localStorage.removeItem('token')
        setUser(null)
    }, [])

    return (
        <div className="terminal">
            <Header user={user} onLogout={handleLogout} />

            <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />

            <main className="terminal__main">
                {activeTab === 'pricing' && <PricingPanel />}
                {activeTab === 'positions' && (
                    <PositionsTable
                        portfolio={activePortfolio}
                        onUpdate={() => fetchPortfolios(localStorage.getItem('token'))}
                    />
                )}
                {activeTab === 'research' && (
                    <div className="card">
                        <div className="card__header">
                            <span className="card__title">Research Dashboard</span>
                        </div>
                        <div className="card__body">
                            <p className="text-muted">Numerical method comparison tools</p>
                        </div>
                    </div>
                )}
            </main>

            <aside className="terminal__panel">
                <OrderForm
                    onLogin={handleLogin}
                    user={user}
                    portfolio={activePortfolio}
                    onOrderPlaced={() => fetchPortfolios(localStorage.getItem('token'))}
                />
            </aside>

            <Footer />
        </div>
    )
}
