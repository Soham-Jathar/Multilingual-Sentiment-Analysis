import { useState } from 'react'

const API = 'http://127.0.0.1:8001/api'
const labels = { positive: ['+', 'Positive', 'positive'], negative: ['-', 'Negative', 'negative'], neutral: ['=', 'Neutral', 'neutral'] }
const languages = [['en', 'English'], ['hi', 'Hindi'], ['mr', 'Marathi'], ['ta', 'Tamil'], ['bn', 'Bengali'], ['te', 'Telugu'], ['gu', 'Gujarati'], ['kn', 'Kannada'], ['ml', 'Malayalam'], ['pa', 'Punjabi'], ['or', 'Odia'], ['as', 'Assamese']]

function Result({ result }) {
  const [icon, label, tone] = labels[result.sentiment]
  return <>
    <section className={`result-card ${tone}`} aria-live="polite"><div className="result-icon">{icon}</div><div><p className="eyebrow">ANALYSIS RESULT</p><h2>{label} sentiment</h2><p className="quoted">&quot;{result.text}&quot;</p></div><div className="metrics"><div><span>Confidence</span><strong>{(result.confidence * 100).toFixed(1)}%</strong></div><div><span>Language</span><strong>{result.language}</strong></div><div><span>Model</span><strong>{result.model.replace('Local TF-IDF + ', '')}</strong></div></div></section>
    {result.translation && <section className="translation-card"><p className="eyebrow">TRANSLATION</p><p>{result.translation}</p><small>Translated from {result.language} to {result.translation_target} using {result.translation_model}</small></section>}
  </>
}

export default function App() {
  const [text, setText] = useState('')
  const [mode, setMode] = useState('transformer')
  const [translationEnabled, setTranslationEnabled] = useState(false)
  const [targetLanguage, setTargetLanguage] = useState('en')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const analyze = async (event) => {
    event.preventDefault()
    if (text.trim().length < 3) return setError('Please enter at least three characters.')
    setLoading(true); setError('')
    try {
      const response = await fetch(`${API}/analyze`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, mode, translation_enabled: translationEnabled, translation_target: targetLanguage }) })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail || 'Analysis could not be completed.')
      setResult(body)
    } catch (requestError) { setError(requestError.message.includes('Failed to fetch') ? 'The analysis service is not running. Start the backend first.' : requestError.message) } finally { setLoading(false) }
  }
  return <main>
    <nav className="nav"><a className="brand" href="#top"><span>SS</span> SentimentScope</a></nav>
    <section className="hero" id="top"><p className="eyebrow">MULTILINGUAL SENTIMENT ANALYSIS</p><h1>Understand every<br /><em>customer voice.</em></h1><p className="lead">Paste a review to identify its sentiment and language. Translation is available when you need it.</p></section>
    <section className="workspace"><form className="analyzer" onSubmit={analyze}><div className="section-title"><div><p className="eyebrow">ANALYZE TEXT</p><h2>What are people saying?</h2></div><span className={`mode-pill ${mode}`}>{mode === 'fast' ? 'Baseline comparison' : 'Fine-tuned XLM-RoBERTa'}</span></div><label htmlFor="feedback">Feedback text</label><textarea id="feedback" value={text} onChange={(event) => setText(event.target.value)} placeholder="Paste a review, comment, or message in any language..." maxLength="2000" /><div className="form-footer"><span>{text.length}/2000 characters</span><label className="switch translation-switch"><input type="checkbox" checked={translationEnabled} onChange={(event) => setTranslationEnabled(event.target.checked)} /><span className="slider" /><b>Translation</b></label>{translationEnabled && <label className="translate-select">Translate to<select value={targetLanguage} onChange={(event) => setTargetLanguage(event.target.value)}>{languages.map(([code, name]) => <option key={code} value={code}>{name}</option>)}</select></label>}<label className="switch"><input type="checkbox" checked={mode === 'transformer'} onChange={(event) => setMode(event.target.checked ? 'transformer' : 'fast')} /><span className="slider" /><b>Use fine-tuned XLM-RoBERTa</b></label><button disabled={loading}>{loading ? (translationEnabled ? 'Translating and analyzing...' : 'Analyzing...') : 'Analyze sentiment'} <span>-&gt;</span></button></div>{error && <p className="error" role="alert">{error}</p>}</form>{result ? <Result result={result} /> : <section className="empty-result"><div className="empty-mark">*</div><p className="eyebrow">READY WHEN YOU ARE</p><h2>Your insight will appear here.</h2><p>Enter feedback above to identify its language and sentiment.</p></section>}</section>
  </main>
}
