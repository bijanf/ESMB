
import { useState, useEffect } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'
import './App.css'

const API_URL = '/api'

function App() {
  const [status, setStatus] = useState('disconnected')
  const [data, setData] = useState(null)
  const [history, setHistory] = useState({
    time: [],
    temp: [],
    amoc: []
  })
  const [isRunning, setIsRunning] = useState(false)

  useEffect(() => {
    const interval = setInterval(fetchData, 500) // Poll every 500ms
    return () => clearInterval(interval)
  }, [])

  const fetchData = async () => {
    try {
      const res = await axios.get(`${API_URL}/state`)
      if (res.data.status === 'not_initialized') {
        setStatus('not_initialized')
      } else {
        setStatus('connected')
        setData(res.data)

        // Update history
        setHistory(prev => {
          // Avoid duplicates
          if (prev.time.length > 0 && prev.time[prev.time.length - 1] === res.data.time) {
            return prev
          }
          const newTime = [...prev.time, res.data.time]
          const newTemp = [...prev.temp, res.data.global_temp]
          const newAmoc = [...prev.amoc, res.data.amoc_index]
          // Keep last 100 points
          if (newTime.length > 100) {
            return {
              time: newTime.slice(-100),
              temp: newTemp.slice(-100),
              amoc: newAmoc.slice(-100)
            }
          }
          return { time: newTime, temp: newTemp, amoc: newAmoc }
        })
      }
    } catch (err) {
      setStatus('disconnected')
    }
  }

  const handleStart = async () => {
    await axios.post(`${API_URL}/start`)
    setIsRunning(true)
  }

  const handleStop = async () => {
    await axios.post(`${API_URL}/stop`)
    setIsRunning(false)
  }

  const handleReset = async () => {
    await axios.post(`${API_URL}/reset`)
    setHistory({ time: [], temp: [] })
    setIsRunning(false)
  }

  const updateParams = async (newParams) => {
    try {
      await axios.post(`${API_URL}/update_params`, newParams)
      // Optimistic update or wait for next poll
    } catch (err) {
      console.error("Failed to update params", err)
    }
  }

  if (status === 'disconnected') {
    return <div className="container"><h1>Connecting to Server...</h1></div>
  }

  return (
    <div className="container">
      <header>
        <h1>Chronos-ESM Dashboard</h1>
        <div className="controls">
          <button onClick={handleStart} disabled={isRunning}>Start</button>
          <button onClick={handleStop} disabled={!isRunning}>Stop</button>
          <button onClick={handleReset}>Reset</button>
          <span className="status">
            Status: {isRunning ? 'Running' : 'Stopped'} |
            Step: {data?.step || 0} |
            Year: {data?.year?.toFixed(2) || '0.00'} |
            AMOC: {data?.amoc_index?.toFixed(2) || '0.00'} Sv
          </span>
        </div>

        {/* Settings Panel */}
        <div className="settings" style={{ marginTop: '10px', padding: '10px', background: '#f5f5f5', borderRadius: '5px', color: '#333' }}>
          <h3>Model Parameters</h3>
          <div style={{ display: 'flex', gap: '20px', alignItems: 'center', justifyContent: 'center', flexWrap: 'wrap' }}>
            <div>
              <label>CO2 (ppm): {data?.params?.co2_ppm?.toFixed(1) || 280}</label>
              <br />
              <input
                type="range"
                min="200"
                max="1000"
                step="10"
                value={data?.params?.co2_ppm || 280}
                onChange={(e) => updateParams({ co2_ppm: parseFloat(e.target.value) })}
              />
            </div>
            <div>
              <label>Solar Constant (W/m²): {data?.params?.solar_constant?.toFixed(1) || 1361}</label>
              <br />
              <input
                type="range"
                min="1300"
                max="1400"
                step="1"
                value={data?.params?.solar_constant || 1361}
                onChange={(e) => updateParams({ solar_constant: parseFloat(e.target.value) })}
              />
            </div>
            <div>
              <label>Target Year: </label>
              <input
                type="number"
                min="0"
                max="10000"
                step="1"
                style={{ width: '80px', padding: '5px' }}
                value={data?.params?.target_year || 2100}
                onChange={(e) => updateParams({ target_year: parseFloat(e.target.value) })}
              />
            </div>
          </div>
        </div>
      </header>

      <div className="grid">
        <div className="card">
          <h3>Global Temperature (K)</h3>
          {data && (
            <Plot
              data={[{
                z: data.fields.temp_atm,
                type: 'heatmap',
                colorscale: 'Viridis'
              }]}
              layout={{
                width: 500,
                height: 300,
                title: `Mean: ${data.global_temp.toFixed(2)} K`,
                margin: { t: 30, b: 30, l: 30, r: 30 }
              }}
            />
          )}
        </div>

        <div className="card">
          <h3>Precipitation (kg/m²/s)</h3>
          <Plot
            data={[{
              z: data?.fields?.precip,
              type: 'heatmap',
              colorscale: 'Blues'
            }]}
            layout={{ width: 500, height: 300, title: 'Precipitation' }}
          />
        </div>

        <div className="card">
          <h3>Vorticity (1/s)</h3>
          <Plot
            data={[{
              z: data?.fields?.vort_atm,
              type: 'heatmap',
              colorscale: 'RdBu'
            }]}
            layout={{ width: 500, height: 300, title: 'Vorticity' }}
          />
        </div>

        <div className="card">
          <h3>CO2 Concentration</h3>
          {data && (
            <Plot
              data={[{
                z: data.fields.co2_atm,
                type: 'heatmap',
                colorscale: 'Hot'
              }]}
              layout={{
                width: 500,
                height: 300,
                title: 'CO2 [ppm]',
                margin: { t: 30, b: 30, l: 30, r: 30 }
              }}
            />
          )}
        </div>

        <div className="card">
          <h3>Global Mean Temperature History</h3>
          <Plot
            data={[{
              x: history.time,
              y: history.temp,
              type: 'scatter',
              mode: 'lines+markers',
              marker: { color: 'orange' }
            }]}
            layout={{
              width: 500,
              height: 300,
              title: 'Global Mean Temp [K]',
              margin: { t: 30, b: 30, l: 30, r: 30 },
              xaxis: { title: 'Time [s]' },
              yaxis: { title: 'Temp [K]' }
            }}
          />
        </div>

        <div className="card">
          <h3>AMOC Index History</h3>
          <Plot
            data={[{
              x: history.time,
              y: history.amoc,
              type: 'scatter',
              mode: 'lines+markers',
              marker: { color: 'purple' }
            }]}
            layout={{
              width: 500,
              height: 300,
              title: 'AMOC Index [Sv]',
              margin: { t: 30, b: 30, l: 30, r: 30 },
              xaxis: { title: 'Time [s]' },
              yaxis: { title: 'AMOC [Sv]' }
            }}
          />
        </div>
      </div>
    </div>
  )
}

export default App
