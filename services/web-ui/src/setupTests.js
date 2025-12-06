// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toBeInTheDocument()
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// Increase default timeout for async operations
jest.setTimeout(10000);

// Configure testing-library
import { configure } from '@testing-library/react';
configure({ asyncUtilTimeout: 5000 });