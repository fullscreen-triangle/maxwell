import { useEffect } from 'react';
import { useRouter } from 'next/router';

export default function ArticlesRedirect() {
  const router = useRouter();
  useEffect(() => { router.replace('/about'); }, [router]);
  return null;
}
