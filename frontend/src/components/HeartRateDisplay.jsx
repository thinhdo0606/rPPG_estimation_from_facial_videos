import { motion } from 'framer-motion'
import { Heart, TrendingDown, TrendingUp, Minus, AlertTriangle } from 'lucide-react'

function HeartRateDisplay({ heartRate, confidence, snr_db, isAnimating }) {
  const beatDuration = heartRate ? 60 / heartRate : 1

  const conf = typeof confidence === 'number' && !Number.isNaN(confidence) ? confidence : null
  const snr = typeof snr_db === 'number' && !Number.isNaN(snr_db) ? snr_db : null

  /**
   * HR bucket (Normal / Elevated / …) should only apply when the PPG is usable.
   * Old logic used confidence >= 0.3 only, so ~33% + negative SNR still showed "Normal" — misleading.
   * Stricter: need decent score AND not strongly noise-dominated spectrum (SNR is in-band spectral, dB).
   */
  const isReliable =
    conf !== null && snr !== null ? conf >= 0.5 && snr >= -2.0 : conf !== null ? conf >= 0.55 : true

  const qualityLabel =
    conf === null || snr === null
      ? null
      : snr < -2 || conf < 0.35
        ? 'Poor'
        : snr < 0 || conf < 0.55
          ? 'Fair'
          : 'Good'

  const getHRStatus = (hr, reliable) => {
    if (!hr) return { label: 'Waiting...', color: 'text-slate-500', icon: Minus, bg: 'bg-slate-900/5' }
    if (!reliable) return { label: 'Poor Signal', color: 'text-rose-700', icon: AlertTriangle, bg: 'bg-rose-500/10' }
    if (hr < 60) return { label: 'Below Normal', color: 'text-blue-700', icon: TrendingDown, bg: 'bg-blue-500/10' }
    if (hr > 100) return { label: 'Elevated', color: 'text-amber-700', icon: TrendingUp, bg: 'bg-amber-500/12' }
    return { label: 'Normal', color: 'text-emerald-700', icon: Minus, bg: 'bg-emerald-500/10' }
  }

  const status = getHRStatus(heartRate, isReliable)
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
                : 'text-slate-300'
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
              <span className="text-xl text-slate-600 ml-1.5">BPM</span>
            </motion.div>
          ) : (
            <div>
              <span className="text-6xl font-bold text-slate-300">--</span>
              <span className="text-xl text-slate-400 ml-1.5">BPM</span>
            </div>
          )}
        </div>

        {/* Status Tag */}
        <div className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium ${status.bg} ${status.color} mb-4`}>
          <StatusIcon className="w-3 h-3" />
          {status.label}
        </div>
        
        {/* Signal Quality Details */}
        {heartRate && conf !== null && snr !== null && (
          <div className="mt-2 text-xs text-slate-500 flex flex-col items-center gap-2 border-t border-slate-100 pt-3">
            {qualityLabel && (
              <span
                className={
                  qualityLabel === 'Poor'
                    ? 'text-rose-600 font-semibold'
                    : qualityLabel === 'Fair'
                      ? 'text-amber-700 font-medium'
                      : 'text-emerald-700 font-medium'
                }
              >
                Signal quality: {qualityLabel}
              </span>
            )}
            <div className="flex justify-center gap-4 w-full">
              <div className="flex flex-col">
                <span className="font-medium text-slate-400 mb-0.5">Quality score</span>
                <span
                  className={
                    qualityLabel === 'Poor'
                      ? 'text-rose-500 font-semibold'
                      : qualityLabel === 'Fair'
                        ? 'text-amber-600 font-semibold'
                        : 'text-emerald-600 font-medium'
                  }
                >
                  {Math.round(conf * 100)}%
                </span>
                <span className="text-[10px] text-slate-400 mt-0.5">From spectral SNR</span>
              </div>
              <div className="w-px bg-slate-200" />
              <div className="flex flex-col">
                <span className="font-medium text-slate-400 mb-0.5">Spectral SNR</span>
                <span
                  className={
                    snr < 0 ? 'text-amber-700 font-semibold' : 'text-slate-700 font-medium'
                  }
                >
                  {snr.toFixed(1)} dB
                </span>
                <span className="text-[10px] text-slate-400 mt-0.5">In-band peak vs rest</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default HeartRateDisplay
