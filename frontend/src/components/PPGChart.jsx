import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, Area, AreaChart } from 'recharts'
import { Activity } from 'lucide-react'

function PPGChart({ data }) {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return []
    const displayData = data.slice(-128)
    return displayData.map((value, index) => ({
      time: index,
      value: value
    }))
  }, [data])

  if (!chartData || chartData.length === 0) {
    return null
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card rounded-2xl p-5"
    >
      <div className="flex items-center gap-2 mb-3">
        <Activity className="w-4 h-4 text-primary-400" />
        <h3 className="font-semibold text-sm">PPG Waveform</h3>
      </div>

      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="ppgGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ff6b6b" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#ff6b6b" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="time"
              tick={false}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: '#4b5563', fontSize: 9 }}
              axisLine={false}
              tickLine={false}
              width={35}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(15, 23, 42, 0.95)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: '10px',
                color: '#fff',
                fontSize: '12px',
                padding: '8px 12px'
              }}
              labelStyle={{ color: '#6b7280', fontSize: '11px' }}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#ff6b6b"
              strokeWidth={1.5}
              fill="url(#ppgGradient)"
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <p className="text-[10px] text-gray-600 mt-2 text-center">
        Photoplethysmogram (PPG) signal extracted from facial video
      </p>
    </motion.div>
  )
}

export default PPGChart
