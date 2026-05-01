import { useEffect, useState } from 'react';

// Tracks which DOM element is most centered in the viewport.
// Used to drive the simulation params on docs pages.
export default function useActiveSection(refs, defaultId = 0) {
  const [activeId, setActiveId] = useState(defaultId);

  useEffect(() => {
    if (!refs || refs.length === 0) return;

    const handleScroll = () => {
      const viewportMid = window.innerHeight / 2;
      let closestIdx = 0;
      let closestDist = Infinity;

      refs.forEach((ref, idx) => {
        if (!ref.current) return;
        const rect = ref.current.getBoundingClientRect();
        const sectionMid = rect.top + rect.height / 2;
        const dist = Math.abs(sectionMid - viewportMid);
        if (dist < closestDist) {
          closestDist = dist;
          closestIdx = idx;
        }
      });

      setActiveId(closestIdx);
    };

    handleScroll();
    window.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('resize', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('resize', handleScroll);
    };
  }, [refs]);

  return activeId;
}
