const PATH_PAGE_KEYS: Array<[string, string]> = [
  ['/analyze', 'analyze'],
  ['/backtest', 'backtest'],
  ['/model', 'model'],
  ['/monitor', 'monitor'],
  ['/scan', 'scan'],
  ['/picks', 'picks'],
  ['/factors', 'factors'],
  ['/risk', 'risk'],
];

export function getPageKeyFromPath(pathname: string): string {
  if (pathname === '/') {
    return 'overview';
  }

  const match = PATH_PAGE_KEYS.find(([prefix]) => pathname.startsWith(prefix));
  return match ? match[1] : 'overview';
}
