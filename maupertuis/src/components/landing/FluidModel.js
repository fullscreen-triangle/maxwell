import { useRef, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { useGLTF, useAnimations } from '@react-three/drei';

export default function FluidModel(props) {
  const group = useRef();
  const { scene, animations } = useGLTF('/physics1_fluid.glb');
  const { actions, names } = useAnimations(animations, group);

  useEffect(() => {
    // Play the first animation if available
    if (names.length > 0 && actions[names[0]]) {
      actions[names[0]].reset().fadeIn(0.5).play();
    }
    return () => {
      names.forEach((name) => {
        if (actions[name]) actions[name].fadeOut(0.5);
      });
    };
  }, [actions, names]);

  useFrame((state) => {
    if (group.current) {
      // Gentle auto-rotation
      group.current.rotation.y = state.clock.elapsedTime * 0.15;
    }
  });

  return (
    <group ref={group} {...props} dispose={null}>
      <primitive object={scene} />
    </group>
  );
}

useGLTF.preload('/physics1_fluid.glb');
