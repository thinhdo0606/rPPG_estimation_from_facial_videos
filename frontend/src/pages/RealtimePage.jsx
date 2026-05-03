import { useState, useRef, useCallback, useEffect } from 'react'
import Webcam from 'react-webcam'
import { motion, AnimatePresence } from 'framer-motion'
import { Heart, Camera, RefreshCw, CheckCircle, AlertCircle, Timer, Clock, Play, Square } from 'lucide-react'
import HeartRateDisplay from '../components/HeartRateDisplay'
import PPGChart from '../components/PPGChart'

const API_BASE = '/api'
const CAPTURE_FPS = 30
const CAPTURE_INTERVAL = Math.round(1000 / CAPTURE_FPS)
const COUNTDOWN_SECONDS = 3

// Duration options (seconds)
const DURATION_OPTIONS = [
  { label: '30s', value: 30 },
  { label: '45s', value: 45 },
  { label: '60s', value: 60 },
]

const STATE = {
  IDLE: 'idle',
  COUNTDOWN: 'countdown',
  CAPTURING: 'capturing',
  PROCESSING: 'processing',
  RESULT: 'result',
  ERROR: 'error'
}

function RealtimePage() {
  const webcamRef = useRef(null)
  const captureIntervalRef = useRef(null)
  const framesRef = useRef([])
  const startTimeRef = useRef(null)

  const [state, setState] = useState(STATE.IDLE)
  const [countdown, setCountdown] = useState(COUNTDOWN_SECONDS)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [isReady, setIsReady] = useState(false)
  const [duration, setDuration] = useState(30)
  const [elapsed, setElapsed] = useState(0)
  const [frameCount, setFrameCount] = useState(0)

  // Countdown before capture
  useEffect(() => {
    if (state === STATE.COUNTDOWN && countdown > 0) {
      const timer = setTimeout(() => setCountdown(prev => prev - 1), 1000)
      return () => clearTimeout(timer)
    } else if (state === STATE.COUNTDOWN && countdown === 0) {
      startCapturing()
    }
  }, [state, countdown])

  // Elapsed time tracker during capture
  useEffect(() => {
    if (state !== STATE.CAPTURING) return
    const timer = setInterval(() => {
      const now = Date.now()
      const sec = Math.floor((now - startTimeRef.current) / 1000)
      setElapsed(sec)
      if (sec >= duration) {
        stopCapturing()
        processFrames()
      }
    }, 250)
    return () => clearInterval(timer)
  }, [state, duration])

  const startMeasurement = () => {
    setError(null)
    setResult(null)
    framesRef.current = []
    setFrameCount(0)
    setElapsed(0)
    setState(STATE.COUNTDOWN)
    setCountdown(COUNTDOWN_SECONDS)
  }

  const captureFrame = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot()
      if (imageSrc) {
        framesRef.current.push(imageSrc)
        setFrameCount(framesRef.current.length)
      }
    }
  }, [])

  const startCapturing = () => {
    framesRef.current = []
    setFrameCount(0)
    setElapsed(0)
    startTimeRef.current = Date.now()
    setState(STATE.CAPTURING)
    captureIntervalRef.current = setInterval(captureFrame, CAPTURE_INTERVAL)
  }

  const stopCapturing = () => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current)
      captureIntervalRef.current = null
    }
  }

  const cancelMeasurement = () => {
    stopCapturing()
    framesRef.current = []
    setFrameCount(0)
    setElapsed(0)
    setState(STATE.IDLE)
  }

  const processFrames = async () => {
    stopCapturing()
    setState(STATE.PROCESSING)
    const capturedFrames = framesRef.current

    try {
      const response = await fetch(`${API_BASE}/predict/realtime`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frames: capturedFrames,
          fps: CAPTURE_FPS
        })
      })

      const data = await response.json()

      if (data.success) {
        setResult({
          heartRate: Math.round(data.heart_rate),
          confidence: data.confidence,
          snr_db: data.snr_db,
          ppgSignal: data.ppg_signal || [],
          totalFrames: capturedFrames.length,
          duration: duration
        })
        setState(STATE.RESULT)
      } else {
        throw new Error(data.message || 'Failed to estimate heart rate')
      }
    } catch (err) {
      setError(err.message)
      setState(STATE.ERROR)
    }
  }

  const reset = () => {
    stopCapturing()
    framesRef.current = []
    setFrameCount(0)
    setResult(null)
    setError(null)
    setElapsed(0)
    setState(STATE.IDLE)
    setCountdown(COUNTDOWN_SECONDS)
  }

  useEffect(() => {
    return () => stopCapturing()
  }, [])

  const remaining = duration - elapsed
  const progress = duration > 0 ? (elapsed / duration) * 100 : 0

  // SVG circular progress
  const circleRadius = 54
  const circleCircumference = 2 * Math.PI * circleRadius
  const circleOffset = circleCircumference - (progress / 100) * circleCircumference

  return (
    <div className="max-w-5xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-6"
      >
        <h1 className="text-3xl font-bold mb-2">
          <span className="gradient-text">Real-time</span> Heart Rate
        </h1>
        <p className="text-slate-600">
          Position your face in the guide and stay still during measurement
        </p>
      </motion.div>

      <div className="grid lg:grid-cols-5 gap-6">
        {/* Left: Webcam (3 cols) */}
        <div className="lg:col-span-3 space-y-4">
          <div className="webcam-wrapper">
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              screenshotQuality={0.8}
              mirrored={true}
              className="w-full h-full object-cover rounded-2xl"
              videoConstraints={{
                width: 640,
                height: 480,
                facingMode: 'user'
              }}
              onUserMedia={() => setIsReady(true)}
            />

            {/* Face Oval Guide */}
            <div className={`face-guide ${
              state === STATE.CAPTURING ? 'face-guide--active' :
              state === STATE.COUNTDOWN ? 'face-guide--ready' :
              state === STATE.RESULT ? 'face-guide--done' : ''
            }`}>
              <div className="face-guide__oval" />
              {state === STATE.CAPTURING && (
                <>
                  <div className="face-guide__pulse" />
                  <div className="face-guide__pulse face-guide__pulse--delayed" />
                </>
              )}
            </div>

            {/* Countdown Overlay */}
            <AnimatePresence>
              {state === STATE.COUNTDOWN && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 flex items-center justify-center bg-black/40 rounded-2xl"
                >
                  <div className="text-center">
                    <motion.div
                      key={countdown}
                      initial={{ scale: 2, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      exit={{ scale: 0.5, opacity: 0 }}
                      transition={{ type: 'spring', stiffness: 200, damping: 15 }}
                      className="text-8xl font-bold text-white drop-shadow-lg mb-2"
                    >
                      {countdown}
                    </motion.div>
                    <p className="text-white/80 text-lg font-medium">Get ready, hold still...</p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Capturing Timer Overlay */}
            <AnimatePresence>
              {state === STATE.CAPTURING && (
                <motion.div
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="absolute top-4 left-1/2 -translate-x-1/2"
                >
                  <div className="capture-timer-badge">
                    <div className="capture-timer-dot" />
                    <span className="font-semibold text-white text-sm">
                      {remaining > 0 ? `${remaining}s remaining` : 'Finishing...'}
                    </span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Processing Overlay */}
            <AnimatePresence>
              {state === STATE.PROCESSING && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 flex items-center justify-center bg-black/60 rounded-2xl"
                >
                  <div className="text-center">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
                      className="mb-4"
                    >
                      <Heart className="w-14 h-14 text-primary-400 fill-primary-400 mx-auto" />
                    </motion.div>
                    <p className="text-white font-semibold text-lg">Analyzing heart rate...</p>
                    <p className="text-white/60 text-sm mt-1">{frameCount} frames captured</p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Status Badge */}
            <div className="absolute top-4 left-4">
              <div className={`status-badge ${
                state === STATE.IDLE ? 'status-badge--idle' :
                state === STATE.COUNTDOWN ? 'status-badge--countdown' :
                state === STATE.CAPTURING ? 'status-badge--capturing' :
                state === STATE.PROCESSING ? 'status-badge--processing' :
                state === STATE.RESULT ? 'status-badge--done' :
                state === STATE.ERROR ? 'status-badge--error' : ''
              }`}>
                {state === STATE.IDLE && <><Camera className="w-4 h-4" /> Ready</>}
                {state === STATE.COUNTDOWN && <><Timer className="w-4 h-4" /> Get Ready</>}
                {state === STATE.CAPTURING && <><div className="w-2 h-2 rounded-full bg-red-400 animate-pulse" /> Recording</>}
                {state === STATE.PROCESSING && <><RefreshCw className="w-4 h-4 animate-spin" /> Processing</>}
                {state === STATE.RESULT && <><CheckCircle className="w-4 h-4" /> Complete</>}
                {state === STATE.ERROR && <><AlertCircle className="w-4 h-4" /> Error</>}
              </div>
            </div>
          </div>

          {/* Capture Progress Bar */}
          {state === STATE.CAPTURING && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-2"
            >
              <div className="flex justify-between text-sm text-slate-600">
                <span>{elapsed}s / {duration}s</span>
                <span>{frameCount} frames</span>
              </div>
              <div className="progress-bar">
                <motion.div
                  className="progress-fill"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </motion.div>
          )}

          {/* Controls */}
          <div className="flex gap-3">
            {(state === STATE.IDLE || state === STATE.ERROR) && (
              <button
                onClick={startMeasurement}
                disabled={!isReady}
                className="btn-primary flex-1 flex items-center justify-center gap-2 text-lg py-4"
              >
                <Play className="w-5 h-5" />
                Start Measurement
              </button>
            )}

            {(state === STATE.CAPTURING || state === STATE.COUNTDOWN) && (
              <button
                onClick={cancelMeasurement}
                className="btn-danger flex-1 flex items-center justify-center gap-2"
              >
                <Square className="w-5 h-5" />
                Cancel
              </button>
            )}

            {(state === STATE.RESULT || state === STATE.ERROR) && (
              <button
                onClick={reset}
                className="btn-secondary flex-1 flex items-center justify-center gap-2"
              >
                <RefreshCw className="w-5 h-5" />
                Measure Again
              </button>
            )}
          </div>

          {/* Error */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-700"
            >
              <div className="flex items-center gap-2 mb-1">
                <AlertCircle className="w-5 h-5" />
                <span className="font-medium">Error</span>
              </div>
              <p className="text-sm">{error}</p>
            </motion.div>
          )}
        </div>

        {/* Right: Controls & Results (2 cols) */}
        <div className="lg:col-span-2 space-y-5">

          {/* Duration Selector */}
          {(state === STATE.IDLE || state === STATE.ERROR) && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-card rounded-2xl p-5"
            >
              <div className="flex items-center gap-2 mb-3">
                <Clock className="w-5 h-5 text-primary-400" />
                <h3 className="font-semibold">Measurement Duration</h3>
              </div>
              <div className="grid grid-cols-3 gap-2">
                {DURATION_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => setDuration(opt.value)}
                    className={`py-3 rounded-xl text-sm font-semibold transition-all duration-200 ${
                      duration === opt.value
                        ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/30'
                        : 'bg-slate-900/5 text-slate-600 hover:bg-slate-900/10 hover:text-slate-900'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
              <p className="text-xs text-slate-500 mt-2 text-center">
                Longer duration = more accurate results
              </p>
            </motion.div>
          )}

          {/* Circular Timer during capture */}
          {state === STATE.CAPTURING && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="glass-card rounded-2xl p-6 flex flex-col items-center"
            >
              <div className="relative w-32 h-32">
                <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
                  <circle
                    cx="60" cy="60" r={circleRadius}
                    fill="none"
                    stroke="rgba(15, 23, 42, 0.08)"
                    strokeWidth="8"
                  />
                  <circle
                    cx="60" cy="60" r={circleRadius}
                    fill="none"
                    stroke="url(#timerGradient)"
                    strokeWidth="8"
                    strokeLinecap="round"
                    strokeDasharray={circleCircumference}
                    strokeDashoffset={circleOffset}
                    className="transition-all duration-500 ease-linear"
                  />
                  <defs>
                    <linearGradient id="timerGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#4ade80" />
                      <stop offset="100%" stopColor="#22d3ee" />
                    </linearGradient>
                  </defs>
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-3xl font-bold text-white">{remaining}</span>
                  <span className="text-xs text-white/80">seconds</span>
                </div>
              </div>
              <p className="text-white/85 text-sm mt-3 font-medium">Keep still, measuring...</p>
              <div className="flex items-center gap-2 mt-2">
                <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                <span className="text-green-400 text-xs font-medium">{frameCount} frames</span>
              </div>
            </motion.div>
          )}

          {/* Heart Rate Display */}
          <HeartRateDisplay
            heartRate={result?.heartRate}
            confidence={result?.confidence}
            snr_db={result?.snr_db}
            isAnimating={state === STATE.RESULT}
          />

          {/* PPG Chart */}
          {result?.ppgSignal && result.ppgSignal.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <PPGChart data={result.ppgSignal} />
            </motion.div>
          )}

          {/* Measurement Info */}
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="glass-card rounded-2xl p-5"
            >
              <h3 className="font-semibold mb-3 text-sm text-slate-700">Measurement Details</h3>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-slate-900/5 rounded-lg p-3">
                  <p className="text-gray-500 text-xs">Duration</p>
                  <p className="font-semibold">{result.duration}s</p>
                </div>
                <div className="bg-slate-900/5 rounded-lg p-3">
                  <p className="text-gray-500 text-xs">Frames</p>
                  <p className="font-semibold">{result.totalFrames}</p>
                </div>
              </div>
            </motion.div>
          )}

          {/* Instructions when idle */}
          {state === STATE.IDLE && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="glass-card rounded-2xl p-5"
            >
              <h3 className="font-semibold mb-3 flex items-center gap-2 text-sm">
                <CheckCircle className="w-4 h-4 text-primary-400" />
                How to Measure
              </h3>
              <ul className="space-y-2.5 text-slate-600 text-sm">
                <li className="flex items-start gap-2">
                  <span className="w-5 h-5 rounded-full bg-primary-500/20 text-primary-400 text-xs flex items-center justify-center shrink-0 mt-0.5">1</span>
                  Position your face inside the green oval guide
                </li>
                <li className="flex items-start gap-2">
                  <span className="w-5 h-5 rounded-full bg-primary-500/20 text-primary-400 text-xs flex items-center justify-center shrink-0 mt-0.5">2</span>
                  Ensure good, even lighting on your face
                </li>
                <li className="flex items-start gap-2">
                  <span className="w-5 h-5 rounded-full bg-primary-500/20 text-primary-400 text-xs flex items-center justify-center shrink-0 mt-0.5">3</span>
                  Select duration and click "Start Measurement"
                </li>
                <li className="flex items-start gap-2">
                  <span className="w-5 h-5 rounded-full bg-primary-500/20 text-primary-400 text-xs flex items-center justify-center shrink-0 mt-0.5">4</span>
                  Stay completely still during measurement
                </li>
                <li className="flex items-start gap-2">
                  <span className="w-5 h-5 rounded-full bg-primary-500/20 text-primary-400 text-xs flex items-center justify-center shrink-0 mt-0.5">5</span>
                  View your heart rate results
                </li>
              </ul>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  )
}

export default RealtimePage
