export const getErrorMessage = (error, fallbackMessage) => {
  if (!error) {
    return fallbackMessage;
  }

  if (error.request && !error.response) {
    return 'Cannot connect to server. Please ensure all services are running and try again.';
  }

  if (error.response) {
    const { status, data } = error.response;
    const serverMessage = data?.detail || data?.message || data?.error;

    switch (status) {
      case 400:
        return serverMessage || 'Invalid request. Please check your input and try again.';
      case 401:
        return 'Authentication required. Please log in and try again.';
      case 403:
        return 'You do not have permission to perform this action.';
      case 404:
        return serverMessage || 'The requested resource was not found.';
      case 422:
        return serverMessage || 'Validation failed. Please check your input.';
      case 500:
        return 'Server error occurred. Please try again later.';
      case 503:
        return 'Service temporarily unavailable. Please try again later.';
      default:
        return serverMessage || fallbackMessage;
    }
  }

  if (error.message) {
    return `Error: ${error.message}`;
  }

  return fallbackMessage;
};

export default getErrorMessage;
