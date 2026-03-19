import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import AppHeader from '@/components/layout/AppHeader';
import AppSidebar from '@/components/layout/AppSidebar';
import Toast from '@/components/ui/Toast';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'QuantTool - 量化交易平台',
  description: '专业量化交易分析平台，提供股票分析、策略回测、模型训练等功能',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN" className="dark">
      <body className={`${inter.className} bg-bg-primary text-text-primary antialiased`}>
        <div className="h-screen flex flex-col">
          <AppHeader />
          <div className="flex-1 flex overflow-hidden">
            <AppSidebar />
            {children}
          </div>
        </div>
        <Toast />
      </body>
    </html>
  );
}
