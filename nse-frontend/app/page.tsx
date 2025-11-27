'use client';

import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, Sun, Moon, RefreshCw } from 'lucide-react';

// --- TYPES ---
type Message = {
  role: 'user' | 'assistant';
  content: string;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', content: 'Hello! I am the NSE Digital Assistant.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // --- SCROLL TO BOTTOM ---
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // --- INITIAL BACKGROUND SYNC ---
  useEffect(() => {
    const syncBackend = async () => {
      try {
        // Trigger backend refresh silently
        await axios.post(`${process.env.NEXT_PUBLIC_API_URL}/refresh`);
        console.log('Background sync triggered.');
      } catch (e) {
        console.warn('Backend sync failed (might be waking up).', e);
      }
    };
    syncBackend();
  }, []);

  // --- HANDLE SEND ---
  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg: Message = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL}/ask`, {
        query: userMsg.content,
      });

      const data = res.data;
      let botText = data.answer || "I couldn't get a response.";

      // Format sources
      if (data.sources && data.sources.length > 0) {
        const sourceLinks = data.sources
          .map((s: string) => {
            const name = s.replace('https://www.nse.co.ke', 'nse.co.ke').split('/').pop() || 'Source';
            return `• [${name}](${s})`;
          })
          .join('\n');
        botText += `\n\n**Sources:**\n${sourceLinks}`;
      }

      setMessages((prev) => [...prev, { role: 'assistant', content: botText }]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: "⚠️ Error: Unable to reach the NSE Engine. Please try again." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // --- RENDER ---
  return (
    <div className={`min-h-screen flex flex-col ${darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-800'} font-sans transition-colors duration-300`}>
      
      {/* BACKGROUND IMAGE OVERLAY */}
      <div 
        className="fixed inset-0 z-0 pointer-events-none opacity-10"
        style={{
          backgroundImage: `url('https://i.postimg.cc/vBh5LSLT/logo.webp')`,
          backgroundRepeat: 'no-repeat',
          backgroundPosition: 'center',
          backgroundSize: '50%'
        }}
      />

      {/* HEADER */}
      <header className="z-10 p-6 text-center border-b border-opacity-10 border-gray-500 bg-opacity-90 backdrop-blur-md sticky top-0">
        <div className="flex justify-between items-center max-w-3xl mx-auto">
          <div className="flex items-center gap-3">
            <img src="https://i.postimg.cc/NF1qzmFV/nse-small-logo.png" alt="NSE Logo" className="w-10 h-10" />
            <div className="text-left">
              <h1 className="text-xl font-bold text-[#0F4C81]">NAIROBI SECURITIES EXCHANGE</h1>
              <p className="text-xs text-gray-500 uppercase tracking-widest">Digital Assistant</p>
            </div>
          </div>
          <button 
            onClick={() => setDarkMode(!darkMode)}
            className={`p-2 rounded-full ${darkMode ? 'bg-gray-800 hover:bg-gray-700' : 'bg-white hover:bg-gray-100'} shadow-sm transition`}
          >
            {darkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </div>
      </header>

      {/* CHAT AREA */}
      <main className="flex-1 z-10 w-full max-w-3xl mx-auto p-4 pb-24 overflow-y-auto">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex gap-3 mb-6 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
            
            {/* Avatar */}
            <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${msg.role === 'assistant' ? 'bg-white border border-gray-200' : 'bg-[#0F4C81]'}`}>
              {msg.role === 'assistant' ? (
                <img src="https://i.postimg.cc/NF1qzmFV/nse-small-logo.png" className="w-6 h-6" />
              ) : (
                <span className="text-white text-xs">User</span>
              )}
            </div>

            {/* Message Bubble */}
            <div className={`p-4 rounded-2xl shadow-sm max-w-[80%] whitespace-pre-wrap leading-relaxed ${
              msg.role === 'user' 
                ? 'bg-[#0F4C81] text-white rounded-tr-none' 
                : (darkMode ? 'bg-gray-800 border-l-4 border-[#4CAF50]' : 'bg-white border-l-4 border-[#4CAF50]')
            }`}>
              {/* Simple Markdown Parser Replacement for Display */}
              {msg.content.split('\n').map((line, i) => (
                <p key={i} className="mb-1">
                  {line.startsWith('•') || line.startsWith('-') ? (
                    <span className="ml-4 block">{line}</span>
                  ) : (
                    line
                  )}
                </p>
              ))}
            </div>
          </div>
        ))}
        
        {loading && (
          <div className="flex items-center gap-2 text-gray-400 ml-12">
            <RefreshCw className="animate-spin w-4 h-4" />
            <span className="text-xs">Analyzing market data...</span>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </main>

      {/* INPUT AREA */}
      <footer className="fixed bottom-0 left-0 w-full z-20 p-4 bg-opacity-90 backdrop-blur-md">
        <div className="max-w-3xl mx-auto relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask anything about the market..."
            className={`w-full pl-6 pr-12 py-4 rounded-full shadow-lg border-2 focus:outline-none focus:border-[#4CAF50] transition-colors ${
              darkMode 
                ? 'bg-gray-800 border-gray-700 text-white placeholder-gray-500' 
                : 'bg-white border-[#0F4C81] text-gray-800 placeholder-gray-400'
            }`}
          />
          <button 
            onClick={handleSend}
            disabled={loading}
            className="absolute right-3 top-3 p-2 bg-[#0F4C81] text-white rounded-full hover:bg-blue-700 transition disabled:opacity-50"
          >
            <Send size={18} />
          </button>
        </div>
        <p className="text-center text-[10px] text-gray-400 mt-2">
          Powered by NSE Engine • Not financial advice
        </p>
      </footer>

    </div>
  );
}