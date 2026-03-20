'use client';

import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '@/components/ui/Table';
import Badge from '@/components/ui/Badge';
import type { TradeRecord } from '@/types/backtest';
import { formatDate, formatNumber, formatAmount } from '@/lib/utils';

interface TradeHistoryProps {
  trades: TradeRecord[];
  maxRows?: number;
}

export default function TradeHistory({ trades, maxRows = 50 }: TradeHistoryProps) {
  const displayTrades = trades.slice(0, maxRows);

  if (trades.length === 0) {
    return (
      <div className="text-center py-8 text-text-muted">
        暂无交易记录
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>日期</TableHead>
            <TableHead>类型</TableHead>
            <TableHead className="text-right">价格</TableHead>
            <TableHead className="text-right">数量</TableHead>
            <TableHead className="text-right">金额</TableHead>
            <TableHead className="text-right">手续费</TableHead>
            <TableHead>原因</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {displayTrades.map((trade, index) => (
            <TableRow key={index} hoverable>
              <TableCell>{formatDate(trade.date)}</TableCell>
              <TableCell>
                <Badge variant={trade.type === 'buy' ? 'success' : 'danger'}>
                  {trade.type === 'buy' ? '买入' : '卖出'}
                </Badge>
              </TableCell>
              <TableCell className="text-right">{formatNumber(trade.price)}</TableCell>
              <TableCell className="text-right">{trade.shares.toLocaleString()}</TableCell>
              <TableCell className="text-right">{formatAmount(trade.amount)}</TableCell>
              <TableCell className="text-right">{formatNumber(trade.commission)}</TableCell>
              <TableCell className="text-text-muted text-sm max-w-[200px] truncate">
                {trade.reason || '-'}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      {trades.length > maxRows && (
        <div className="text-center py-3 text-sm text-text-muted">
          显示前 {maxRows} 条，共 {trades.length} 条记录
        </div>
      )}
    </div>
  );
}
