// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toBeInTheDocument()
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// React 18: inform React that we're in an act-compatible test env to avoid legacy warnings
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

// Increase default timeout for async operations
jest.setTimeout(10000);

// Configure testing-library
import { configure } from '@testing-library/react';
configure({ asyncUtilTimeout: 5000 });

// Silence noisy console.log output during tests while keeping warnings/errors visible
const originalConsoleLog = console.log;
beforeAll(() => {
  jest.spyOn(console, 'log').mockImplementation((...args) => {
    if (process.env.DEBUG_LOGS === 'true') {
      originalConsoleLog(...args);
    }
  });
});

afterAll(() => {
  console.log.mockRestore();
});

// Provide a stubbed clipboard API for components that interact with it
beforeAll(() => {
  if (!navigator.clipboard) {
    Object.assign(navigator, {
      clipboard: {
        writeText: jest.fn().mockResolvedValue(undefined),
      },
    });
  } else if (!navigator.clipboard.writeText) {
    navigator.clipboard.writeText = jest.fn().mockResolvedValue(undefined);
  } else {
    jest.spyOn(navigator.clipboard, 'writeText').mockResolvedValue(undefined);
  }
});
