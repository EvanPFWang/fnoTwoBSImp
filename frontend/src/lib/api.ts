import axios from 'axios'

const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL,
  headers: { 'Content-Type': 'application/json' }
})

export type OptionType = 'call' | 'put'

export interface PriceReq {
  s: number; k: number; r: number; q: number; t: number; sigma: number; option_type: OptionType
}

export interface IVReq {
  s: number; k: number; r: number; q: number; t: number; market_price: number; option_type: OptionType
}

export interface PriceResp {
  price: number;
  greeks: Record<string, number>;
  implied_vol?: number | null;
}

export interface CurveReq {
  k: number; r: number; q: number; t: number; sigma: number; option_type: OptionType;
  s_min: number; s_max: number; steps: number
}

export async function price(req: PriceReq) {
  const { data } = await api.post<PriceResp>('/api/price', req)
  return data
}
export async function impliedVol(req: IVReq) {
  const { data } = await api.post<PriceResp>('/api/implied-vol', req)
  return data
}
export async function curve(req: CurveReq) {
  const { data } = await api.post<{s:number[]; price:number[]}>('/api/curve', req)
  return data
}
export async function listCalculations(limit=50) {
  const { data } = await api.get<{items:any[]}>('/api/calculations', { params: { limit } })
  return data.items
}
