import { Link, useLocation } from 'react-router-dom'
import { Heart, Camera, Upload, Home } from 'lucide-react'
import { motion } from 'framer-motion'

const navItems = [
  { path: '/', label: 'Home', icon: Home },
  { path: '/realtime', label: 'Real-time', icon: Camera },
  { path: '/upload', label: 'Upload Video', icon: Upload },
]

function Navbar() {
  const location = useLocation()
  
  return (
    <nav className="sticky top-0 z-50 glass-card border-b border-white/10">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2">
            <motion.div
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            >
              <Heart className="w-8 h-8 text-primary-400 fill-primary-400" />
            </motion.div>
            <span className="text-xl font-bold gradient-text">HeartRate AI</span>
          </Link>
          
          {/* Navigation */}
          <div className="flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = location.pathname === item.path
              
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`
                    flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-300
                    ${isActive 
                      ? 'bg-primary-500/20 text-primary-400' 
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                    }
                  `}
                >
                  <Icon className="w-4 h-4" />
                  <span className="font-medium hidden sm:block">{item.label}</span>
                </Link>
              )
            })}
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar

