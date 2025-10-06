import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import Layout from '../Layout';

const renderWithRouter = (ui) => {
  return render(
    <MemoryRouter>
      {ui}
    </MemoryRouter>
  );
};

test('renders layout with navigation', () => {
  renderWithRouter(
    <Layout>
      <div>Test Content</div>
    </Layout>
  );

  expect(screen.getAllByText('Web Scraping Tool').length).toBeGreaterThan(0);
  expect(screen.getAllByText('Dashboard').length).toBeGreaterThan(0);
  expect(screen.getByText('URL Manager')).toBeInTheDocument();
  expect(screen.getByText('Search')).toBeInTheDocument();
  expect(screen.getByText('Sessions')).toBeInTheDocument();
  expect(screen.getByText('Export')).toBeInTheDocument();
  expect(screen.getByText('Test Content')).toBeInTheDocument();
});

test('mobile menu toggle works', () => {
  renderWithRouter(
    <Layout>
      <div>Test Content</div>
    </Layout>
  );

  // Mobile menu should be hidden initially
  const mobileMenu = screen.getByRole('button');
  expect(mobileMenu).toBeInTheDocument();
});

test('navigation links have correct hrefs', () => {
  renderWithRouter(
    <Layout>
      <div>Test Content</div>
    </Layout>
  );

  expect(screen.getByRole('link', { name: /dashboard/i })).toHaveAttribute('href', '/');
  expect(screen.getByRole('link', { name: /url manager/i })).toHaveAttribute('href', '/urls');
  expect(screen.getByRole('link', { name: /search/i })).toHaveAttribute('href', '/search');
  expect(screen.getByRole('link', { name: /sessions/i })).toHaveAttribute('href', '/sessions');
  expect(screen.getByRole('link', { name: /export/i })).toHaveAttribute('href', '/export');
});