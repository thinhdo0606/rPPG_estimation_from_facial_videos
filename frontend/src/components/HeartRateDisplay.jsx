import { motion } from 'framer-motion'
import { Heart, TrendingDown, TrendingUp, Minus } from 'lucide-react'

function HeartRateDisplay({ heartRate, confidence, isAnimating }) {
  const beatDuration = heartRate ? 60 / heartRate : 1

  const getHRStatus = (hr) => {
    if (!hr) return { label: 'Waiting...', color: 'text-gray-500', icon: Minus, bg: 'bg-gray-500/10' }
    if (hr < 60) return { label: 'Below Normal', color: 'text-blue-400', icon: TrendingDown, bg: 'bg-blue-500/10' }
    if (hr > 100) return { label: 'Elevated', color: 'text-yellow-400', icon: TrendingUp, bg: 'bg-yellow-500/10' }
    return { label: 'Normal', color: 'text-green-400', icon: Minus, bg: 'bg-green-500/10' }
  }

  const status = getHRStatus(heartRate)
  const StatusIcon = status.icon

  return (
    <div className="glass-card rounded-2xl p-6">
      <div className="text-center">
        {/* Heart Icon */}
        <motion.div
          animate={isAnimating && heartRate ? {
            scale: [1, 1.2, 1]
          } : {}}
          transition={{
            duration: beatDuration,
            repeat: Infinity,
            ease: 'easeInOut'
          }}
          className="inline-block mb-3"
        >
          <Heart
            className={`w-12 h-12 ${
              heartRate
                ? 'text-primary-400 fill-primary-400 drop-shadow-lg'
                : 'text-gray-700'
            }`}
            style={heartRate ? { filter: 'drop-shadow(0 0 8px rgba(255, 107, 107, 0.4))' } : {}}
          />
        </motion.div>

        {/* Heart Rate Value */}
        <div className="mb-3">
          {heartRate ? (
            <motion.div
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: 'spring', duration: 0.5 }}
            >
              <span className="text-6xl font-bold gradient-text">{heartRate}</span>
              <span className="text-xl text-gray-400 ml-1.5">BPM</span>
            </motion.div>
          ) : (
            <div>
              <span className="text-6xl font-bold text-gray-700">--</span>
              <span className="text-xl text-gray-700 ml-1.5">BPM</span>
            </div>
          )}
        </div>

        {/* Status Tag */}
        <div className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium ${status.bg} ${status.color} mb-4`}>
          <StatusIcon className="w-3 h-3" />
          {status.label}
        </div>

        {/* Confidence Bar */}
        {confidence !== undefined && confidence !== null && heartRate && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-xs mx-auto"
          >
            <div className="flex justify-between text-xs mb-1.5">
              <span className="text-gray-500">Confidence</span>
              <span className={`font-semibold ${
                confidence >= 70 ? 'text-green-400' :
                confidence >= 40 ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {confidence}%
              </span>
            </div>
            <div className="h-1.5 bg-white/8 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${confidence}%` }}
                transition={{ duration: 0.8, ease: 'easeOut' }}
                className={`h-full rounded-full ${
                  confidence >= 70 ? 'bg-green-400' :
                  confidence >= 40 ? 'bg-yellow-400' :
                  'bg-red-400'
                }`}
              />
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}

export default HeartRateDisplay
