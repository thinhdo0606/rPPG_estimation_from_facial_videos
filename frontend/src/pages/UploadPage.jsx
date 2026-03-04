import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, FileVideo, X, Heart, AlertCircle, CheckCircle, Clock, Layers } from 'lucide-react'
import HeartRateDisplay from '../components/HeartRateDisplay'
import PPGChart from '../components/PPGChart'

const API_BASE = '/api'
const MAX_FILE_SIZE = 100 * 1024 * 1024 // 100MB
const ALLOWED_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm']

function UploadPage() {
  const fileInputRef = useRef(null)
  
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  
  // Handle file selection
  const handleFileSelect = (selectedFile) => {
    if (!selectedFile) return
    
    // Validate file type
    if (!ALLOWED_TYPES.includes(selectedFile.type) && !selectedFile.name.match(/\.(mp4|mov|avi|webm)$/i)) {
      setError('Please select a valid video file (MP4, MOV, AVI, WebM)')
      return
    }
    
    // Validate file size
    if (selectedFile.size > MAX_FILE_SIZE) {
      setError('File is too large. Maximum size is 100MB')
      return
    }
    
    setError(null)
    setResult(null)
    setFile(selectedFile)
    
    // Create preview URL
    const url = URL.createObjectURL(selectedFile)
    setPreview(url)
  }
  
  // Handle drag and drop
  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }
  
  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }
  
  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    
    const droppedFile = e.dataTransfer.files[0]
    handleFileSelect(droppedFile)
  }
  
  // Upload and analyze
  const analyzeVideo = async () => {
    if (!file) return
    
    setIsUploading(true)
    setUploadProgress(0)
    setError(null)
    
    try {
      const formData = new FormData()
      formData.append('video', file)
      
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90))
      }, 200)
      
      const response = await fetch(`${API_BASE}/predict/video`, {
        method: 'POST',
        body: formData
      })
      
      clearInterval(progressInterval)
      setUploadProgress(100)
      
      const data = await response.json()
      
      if (data.success) {
        setResult({
          heartRate: Math.round(data.heart_rate),
          confidence: Math.round(data.confidence * 100),
          ppgSignal: data.ppg_signal || [],
          videoDuration: data.video_duration,
          fps: data.fps,
          totalFrames: data.total_frames,
          processingTime: data.processing_time_ms
        })
      } else {
        throw new Error(data.message || 'Failed to analyze video')
      }
    } catch (err) {
      setError(err.message || 'Failed to connect to server')
    } finally {
      setIsUploading(false)
    }
  }
  
  // Clear selected file
  const clearFile = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }
  
  return (
    <div className="max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <h1 className="text-3xl font-bold mb-2">
          <span className="gradient-text">Upload</span> Video
        </h1>
        <p className="text-gray-400">
          Upload a video of your face for heart rate analysis
        </p>
      </motion.div>
      
      <div className="grid lg:grid-cols-2 gap-8">
        {/* Upload Section */}
        <div className="space-y-6">
          {/* Drop Zone */}
          {!file ? (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className={`
                glass-card rounded-2xl p-8 border-2 border-dashed transition-all duration-300 cursor-pointer
                ${isDragging ? 'border-primary-400 bg-primary-500/10' : 'border-white/20 hover:border-white/40'}
              `}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => handleFileSelect(e.target.files[0])}
              />
              
              <div className="text-center">
                <div className="w-16 h-16 rounded-full bg-primary-500/20 flex items-center justify-center mx-auto mb-4">
                  <Upload className="w-8 h-8 text-primary-400" />
                </div>
                <h3 className="text-lg font-semibold mb-2">
                  Drop your video here
                </h3>
                <p className="text-gray-400 text-sm mb-4">
                  or click to browse
                </p>
                <div className="flex flex-wrap justify-center gap-2 text-xs text-gray-500">
                  <span className="px-2 py-1 bg-white/5 rounded">MP4</span>
                  <span className="px-2 py-1 bg-white/5 rounded">MOV</span>
                  <span className="px-2 py-1 bg-white/5 rounded">AVI</span>
                  <span className="px-2 py-1 bg-white/5 rounded">WebM</span>
                  <span className="px-2 py-1 bg-white/5 rounded">Max 100MB</span>
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="glass-card rounded-2xl overflow-hidden"
            >
              {/* Video Preview */}
              <div className="relative aspect-video bg-black">
                <video
                  src={preview}
                  className="w-full h-full object-contain"
                  controls
                />
                <button
                  onClick={clearFile}
                  className="absolute top-2 right-2 p-2 rounded-full bg-black/50 hover:bg-black/70 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              {/* File Info */}
              <div className="p-4">
                <div className="flex items-center gap-3">
                  <FileVideo className="w-10 h-10 text-primary-400" />
                  <div className="flex-1 min-w-0">
                    <p className="font-medium truncate">{file.name}</p>
                    <p className="text-sm text-gray-400">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
          
          {/* Upload Progress */}
          {isUploading && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-gray-400">
                <span>Processing video...</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="progress-bar">
                <motion.div 
                  className="progress-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}
          
          {/* Analyze Button */}
          {file && !isUploading && (
            <button
              onClick={analyzeVideo}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              <Heart className="w-5 h-5" />
              Analyze Heart Rate
            </button>
          )}
          
          {/* Error Message */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="bg-red-500/20 border border-red-500/50 rounded-xl p-4 text-red-300"
              >
                <div className="flex items-center gap-2 mb-1">
                  <AlertCircle className="w-5 h-5" />
                  <span className="font-medium">Error</span>
                </div>
                <p className="text-sm">{error}</p>
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Tips */}
          <div className="glass-card rounded-2xl p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-primary-400" />
              Tips for Best Results
            </h3>
            <ul className="space-y-2 text-gray-400 text-sm">
              <li>Video should be 5-30 seconds long</li>
              <li>Face should be visible throughout the video</li>
              <li>Good lighting improves accuracy</li>
              <li>Minimal movement gives better results</li>
              <li>Front-facing camera angle is preferred</li>
            </ul>
          </div>
        </div>
        
        {/* Results Section */}
        <div className="space-y-6">
          {/* Heart Rate Display */}
          <HeartRateDisplay 
            heartRate={result?.heartRate}
            confidence={result?.confidence}
            isAnimating={!!result}
          />
          
          {/* Video Stats */}
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-card rounded-2xl p-6"
            >
              <h3 className="font-semibold mb-4">Video Analysis</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                    <Clock className="w-5 h-5 text-blue-400" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">Duration</p>
                    <p className="font-medium">{result.videoDuration?.toFixed(1)}s</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-green-500/20 flex items-center justify-center">
                    <Layers className="w-5 h-5 text-green-400" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">Frames</p>
                    <p className="font-medium">{result.totalFrames}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-yellow-500/20 flex items-center justify-center">
                    <FileVideo className="w-5 h-5 text-yellow-400" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">FPS</p>
                    <p className="font-medium">{result.fps?.toFixed(1)}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
                    <Heart className="w-5 h-5 text-purple-400" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">Processing</p>
                    <p className="font-medium">{(result.processingTime / 1000).toFixed(1)}s</p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
          
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
          
          {/* Placeholder when no result */}
          {!result && !isUploading && (
            <div className="glass-card rounded-2xl p-8 text-center">
              <FileVideo className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-500">
                Upload a video to see heart rate analysis
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default UploadPage

