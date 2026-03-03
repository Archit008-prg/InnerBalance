/**
 * GSAP Animation Utilities
 * Reusable animation functions for consistent animations across the app
 */

import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

// Register ScrollTrigger plugin
if (typeof window !== 'undefined') {
  gsap.registerPlugin(ScrollTrigger);
}

/**
 * Fade in animation
 */
export const fadeIn = (element, delay = 0, duration = 0.8) => {
  return gsap.fromTo(
    element,
    { opacity: 0, y: 30 },
    { opacity: 1, y: 0, duration, delay, ease: 'power2.out' }
  );
};

/**
 * Slide in from left
 */
export const slideInLeft = (element, delay = 0, duration = 0.8) => {
  return gsap.fromTo(
    element,
    { opacity: 0, x: -50 },
    { opacity: 1, x: 0, duration, delay, ease: 'power2.out' }
  );
};

/**
 * Slide in from right
 */
export const slideInRight = (element, delay = 0, duration = 0.8) => {
  return gsap.fromTo(
    element,
    { opacity: 0, x: 50 },
    { opacity: 1, x: 0, duration, delay, ease: 'power2.out' }
  );
};

/**
 * Scale in animation
 */
export const scaleIn = (element, delay = 0, duration = 0.6) => {
  return gsap.fromTo(
    element,
    { opacity: 0, scale: 0.8 },
    { opacity: 1, scale: 1, duration, delay, ease: 'back.out(1.7)' }
  );
};

/**
 * Stagger animation for multiple elements
 */
export const staggerFadeIn = (elements, delay = 0, stagger = 0.1) => {
  return gsap.fromTo(
    elements,
    { opacity: 0, y: 30 },
    {
      opacity: 1,
      y: 0,
      duration: 0.8,
      delay,
      stagger,
      ease: 'power2.out',
    }
  );
};

/**
 * Scroll-triggered animation
 */
export const scrollReveal = (element, options = {}) => {
  const {
    delay = 0,
    duration = 0.8,
    start = 'top 80%',
    end = 'bottom 20%',
    toggleActions = 'play none none none',
  } = options;

  return gsap.fromTo(
    element,
    { opacity: 0, y: 50 },
    {
      opacity: 1,
      y: 0,
      duration,
      delay,
      ease: 'power2.out',
      scrollTrigger: {
        trigger: element,
        start,
        end,
        toggleActions,
        once: true,
      },
    }
  );
};

/**
 * Parallax effect
 */
export const parallax = (element, speed = 0.5) => {
  return gsap.to(element, {
    yPercent: -50 * speed,
    ease: 'none',
    scrollTrigger: {
      trigger: element,
      start: 'top bottom',
      end: 'bottom top',
      scrub: true,
    },
  });
};

/**
 * Hover animation
 */
export const hoverScale = (element, scale = 1.05) => {
  const handleMouseEnter = () => {
    gsap.to(element, { scale, duration: 0.3, ease: 'power2.out' });
  };
  const handleMouseLeave = () => {
    gsap.to(element, { scale: 1, duration: 0.3, ease: 'power2.out' });
  };

  if (element.addEventListener) {
    element.addEventListener('mouseenter', handleMouseEnter);
    element.addEventListener('mouseleave', handleMouseLeave);
  }

  return () => {
    if (element.removeEventListener) {
      element.removeEventListener('mouseenter', handleMouseEnter);
      element.removeEventListener('mouseleave', handleMouseLeave);
    }
  };
};

/**
 * Page transition animation
 */
export const pageTransition = {
  enter: (element) => {
    return gsap.fromTo(
      element,
      { opacity: 0, y: 20 },
      { opacity: 1, y: 0, duration: 0.6, ease: 'power2.out' }
    );
  },
  exit: (element) => {
    return gsap.to(element, {
      opacity: 0,
      y: -20,
      duration: 0.4,
      ease: 'power2.in',
    });
  },
};

/**
 * Cleanup function for animations
 */
export const cleanup = (animations) => {
  if (Array.isArray(animations)) {
    animations.forEach((anim) => anim?.kill?.());
  } else {
    animations?.kill?.();
  }
  ScrollTrigger.getAll().forEach((trigger) => trigger.kill());
};

