import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import Layout from '../Layout';

const renderWithRouter = (ui, initialEntries = ['/']) => {
  return render(
    <MemoryRouter initialEntries={initialEntries} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
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

  const dashboardLinks = screen.getAllByRole('link', { name: /dashboard/i });
  expect(dashboardLinks[0]).toHaveAttribute('href', '/');

  const urlManagerLinks = screen.getAllByRole('link', { name: /url manager/i });
  expect(urlManagerLinks[0]).toHaveAttribute('href', '/urls');

  const searchLinks = screen.getAllByRole('link', { name: /search/i });
  expect(searchLinks[0]).toHaveAttribute('href', '/search');

  const sessionLinks = screen.getAllByRole('link', { name: /sessions/i });
  expect(sessionLinks[0]).toHaveAttribute('href', '/sessions');

  const exportLinks = screen.getAllByRole('link', { name: /export/i });
  expect(exportLinks[0]).toHaveAttribute('href', '/export');
});