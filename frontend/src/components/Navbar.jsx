import { Link, useLocation } from 'react-router-dom'
import { ShieldCheck } from 'lucide-react'

export default function Navbar() {
  const { pathname } = useLocation()

  const link = (to, label) => (
    <Link
      to={to}
      className={`text-sm font-medium transition-colors hover:text-white ${
        pathname === to ? 'text-white' : 'text-gray-400'
      }`}
    >
      {label}
    </Link>
  )

  return (
    <nav className="sticky top-0 z-50 bg-gray-950/80 backdrop-blur-md border-b border-gray-800">
      <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 text-white font-bold text-lg">
          <ShieldCheck className="w-6 h-6 text-brand-500" />
          AV-Forensics
        </Link>
        <div className="flex items-center gap-6">
          {link('/', 'Analyze')}
          {link('/about', 'About')}
          <a
            href="https://github.com/kuladeepb/AV-Forensics-Deepfake-Detection"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-medium text-gray-400 hover:text-white transition-colors"
          >
            GitHub
          </a>
        </div>
      </div>
    </nav>
  )
}

