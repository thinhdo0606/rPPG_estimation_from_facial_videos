import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Camera, Upload, Heart, Zap, Shield, Cpu } from 'lucide-react'

const features = [
  {
    icon: Camera,
    title: 'Real-time Detection',
    description: 'Measure your heart rate instantly using your webcam with face tracking technology'
  },
  {
    icon: Upload,
    title: 'Video Analysis',
    description: 'Upload a video for detailed heart rate analysis with PPG waveform visualization'
  },
  {
    icon: Zap,
    title: 'Fast & Accurate',
    description: 'AI-powered rPPG technology provides quick results with high accuracy'
  },
  {
    icon: Shield,
    title: 'Privacy First',
    description: 'All processing happens locally or on your server. Your data stays private'
  },
]

function HomePage() {
  return (
    <div className="max-w-6xl mx-auto">
      {/* Hero Section */}
      <section className="text-center py-16 md:py-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex justify-center mb-6">
            <motion.div
              animate={{ scale: [1, 1.15, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
              className="p-4 rounded-full bg-primary-500/20"
            >
              <Heart className="w-16 h-16 text-primary-400 fill-primary-400" />
            </motion.div>
          </div>
          
          <h1 className="text-4xl md:text-6xl font-bold mb-6">
            <span className="gradient-text">Heart Rate</span> Monitor
          </h1>
          
          <p className="text-xl text-slate-600 max-w-2xl mx-auto mb-10">
            Measure your heart rate contactlessly using AI-powered facial analysis. 
            No wearables needed - just your camera.
          </p>
          
          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/realtime" className="btn-primary flex items-center gap-2 justify-center">
              <Camera className="w-5 h-5" />
              Start Real-time
            </Link>
            <Link to="/upload" className="btn-secondary flex items-center gap-2 justify-center">
              <Upload className="w-5 h-5" />
              Upload Video
            </Link>
          </div>
        </motion.div>
      </section>
      
      {/* Features Section */}
      <section className="py-16">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.5 }}
        >
          <h2 className="text-3xl font-bold text-center mb-12">
            How It <span className="gradient-text">Works</span>
          </h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => {
              const Icon = feature.icon
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 + index * 0.1 }}
                  className="glass-card p-6 rounded-2xl hover:bg-slate-900/5 transition-all duration-300"
                >
                  <div className="w-12 h-12 rounded-xl bg-primary-500/20 flex items-center justify-center mb-4">
                    <Icon className="w-6 h-6 text-primary-400" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                  <p className="text-slate-600 text-sm">{feature.description}</p>
                </motion.div>
              )
            })}
          </div>
        </motion.div>
      </section>
      
      {/* Technology Section */}
      <section className="py-16">
        <div className="glass-card rounded-3xl p-8 md:p-12">
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div>
              <div className="flex items-center gap-2 text-primary-400 mb-4">
                <Cpu className="w-5 h-5" />
                <span className="font-medium">About this website</span>
              </div>
              <h2 className="text-3xl font-bold mb-4">
                Contactless Heart Rate Monitoring
              </h2>
              <p className="text-slate-600 mb-6">
                This website is a research-oriented prototype for estimating heart rate from facial video.
                The goal is to provide a simple, accessible way to monitor heart rate without wearables by using only a camera.
                You can measure in real time with guided face alignment, or upload a short video for analysis and waveform visualization.
              </p>
              <div className="flex flex-wrap gap-3">
                {['Remote monitoring', 'Camera-based', 'Privacy-aware', 'Research prototype'].map((tag) => (
                  <span 
                    key={tag}
                    className="px-3 py-1 rounded-full bg-primary-500/20 text-primary-400 text-sm"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
            <div className="relative">
              <div className="aspect-video bg-gradient-to-br from-primary-500/20 to-primary-400/10 rounded-2xl flex items-center justify-center">
                <motion.div
                  animate={{ 
                    scale: [1, 1.05, 1],
                    rotate: [0, 5, -5, 0]
                  }}
                  transition={{ duration: 4, repeat: Infinity }}
                  className="text-6xl"
                >
                  <Heart className="w-24 h-24 text-primary-400 fill-primary-400" />
                </motion.div>
              </div>
            </div>
          </div>
        </div>
      </section>
      
      {/* Instructions */}
      <section className="py-16">
        <h2 className="text-3xl font-bold text-center mb-12">
          Getting <span className="gradient-text">Started</span>
        </h2>
        
        <div className="grid md:grid-cols-3 gap-6">
          {[
            { step: '01', title: 'Position Your Face', desc: 'Align your face within the oval guide on screen' },
            { step: '02', title: 'Stay Still', desc: 'Keep still for 30-60 seconds while we measure' },
            { step: '03', title: 'Get Results', desc: 'View your heart rate and PPG waveform instantly' },
          ].map((item, index) => (
            <motion.div
              key={item.step}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 + index * 0.1 }}
              className="text-center"
            >
              <div className="text-5xl font-bold gradient-text mb-4">{item.step}</div>
              <h3 className="text-xl font-semibold mb-2">{item.title}</h3>
              <p className="text-slate-600">{item.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>
    </div>
  )
}

export default HomePage

