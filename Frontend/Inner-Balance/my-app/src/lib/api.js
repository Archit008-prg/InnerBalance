/**
 * API Client for Inner Balance
 * Handles all backend API calls with proper error handling, retries, and fallbacks
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000/api';
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

// Default fallback questions if API fails
const FALLBACK_QUESTIONS = [
  {
    id: 1,
    text: "How often have you felt little interest or pleasure in doing things over the past two weeks?",
    question_type: "scale",
    type: "scale",
    category: "depression",
    order: 1
  },
  {
    id: 2,
    text: "How often have you felt down, depressed, or hopeless during the last two weeks?",
    question_type: "scale",
    type: "scale",
    category: "depression",
    order: 2
  },
  {
    id: 3,
    text: "How often have you had trouble falling asleep, staying asleep, or sleeping too much?",
    question_type: "scale",
    type: "scale",
    category: "sleep",
    order: 3
  },
  {
    id: 4,
    text: "How often have you felt tired or had little energy, even after rest?",
    question_type: "scale",
    type: "scale",
    category: "energy",
    order: 4
  },
  {
    id: 5,
    text: "How often have you experienced poor appetite or overeating in the past two weeks?",
    question_type: "scale",
    type: "scale",
    category: "appetite",
    order: 5
  },
  {
    id: 6,
    text: "How often have you felt nervous, anxious, or on edge?",
    question_type: "scale",
    type: "scale",
    category: "anxiety",
    order: 6
  },
  {
    id: 7,
    text: "How often have you been unable to stop or control worrying?",
    question_type: "scale",
    type: "scale",
    category: "anxiety",
    order: 7
  },
  {
    id: 8,
    text: "How often have you felt so restless that it is hard to sit still?",
    question_type: "scale",
    type: "scale",
    category: "anxiety",
    order: 8
  },
  {
    id: 9,
    text: "How often have you become easily annoyed or irritable?",
    question_type: "scale",
    type: "scale",
    category: "anxiety",
    order: 9
  },
  {
    id: 10,
    text: "How often have you felt afraid as if something awful might happen?",
    question_type: "scale",
    type: "scale",
    category: "anxiety",
    order: 10
  },
  {
    id: 11,
    text: "How often have you felt bad about yourself - or that you are a failure or have let yourself or your family down?",
    question_type: "scale",
    type: "scale",
    category: "depression",
    order: 11
  },
  {
    id: 12,
    text: "How often have you had trouble concentrating on things, such as reading the newspaper or watching television?",
    question_type: "scale",
    type: "scale",
    category: "depression",
    order: 12
  },
  {
    id: 13,
    text: "How often have you moved or spoken so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?",
    question_type: "scale",
    type: "scale",
    category: "depression",
    order: 13
  },
  {
    id: 14,
    text: "How often have you worried too much about different things?",
    question_type: "scale",
    type: "scale",
    category: "anxiety",
    order: 14
  },
  {
    id: 15,
    text: "How often have you had trouble relaxing?",
    question_type: "scale",
    type: "scale",
    category: "anxiety",
    order: 15
  },
  {
    id: 16,
    text: "On a scale of 1.0 to 5.0, how intense has your anxiety or worry been when at its worst over the past two weeks?",
    question_type: "slider",
    type: "slider",
    category: "anxiety",
    order: 16
  },
  {
    id: 17,
    text: "On a scale of 1.0 to 5.0, how intense has your low mood or lack of interest been when at its worst over the past two weeks?",
    question_type: "slider",
    type: "slider",
    category: "depression",
    order: 17
  },
  {
    id: 18,
    text: "If you checked off any problems, how difficult have these problems made it for you to do your work, take care of things at home, or get along with other people?",
    question_type: "scale",
    type: "scale",
    category: "functioning",
    order: 18
  }
];

/**
 * Check if backend server is reachable
 */
async function checkBackendHealth() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000); // 3 second timeout
    
    const response = await fetch(`${API_URL}/api/test/`, {
      method: 'GET',
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    clearTimeout(timeoutId);
    return response.ok;
  } catch (error) {
    console.warn('Backend health check failed:', error.message);
    return false;
  }
}

/**
 * Generic API request handler with retry logic and timeout
 * 
 * Note: Network errors are expected when the backend server is not running.
 * These errors are caught and handled gracefully by calling functions,
 * which will use fallback data. The console may show these errors, but
 * the application will continue to function normally.
 */
async function apiRequest(endpoint, options = {}, retries = 2) {
  const url = `${API_BASE_URL}${endpoint}`;
  
  // Create abort controller for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 35000); // 35 second timeout
  
  // Skip sending JWT headers for public endpoints to prevent false-positives
  const publicEndpoints = ['/questions/', '/test/', '/system-status/'];
  const isPublic = publicEndpoints.some(ep => endpoint.endsWith(ep));
  
  // Load token from localStorage if in browser environment and endpoint is not public
  const token = !isPublic && typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
  const authHeader = token ? { 'Authorization': `Bearer ${token}` } : {};
  
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...authHeader,
      ...options.headers,
    },
    signal: controller.signal,
    ...options,
  };

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url, config);
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = errorData.error || errorData.detail || `HTTP error! status: ${response.status}`;
        
        // Handle token expiration / unauthorized error by clearing session and redirecting
        if (response.status === 401 && typeof window !== 'undefined') {
          console.warn('Session expired or invalid token. Redirecting to login...');
          localStorage.removeItem('access_token');
          localStorage.removeItem('refresh_token');
          localStorage.removeItem('user_role');
          localStorage.removeItem('user_name');
          window.location.href = '/login';
          return;
        }
        
        // Don't retry on 4xx errors (client errors)
        if (response.status >= 400 && response.status < 500) {
          throw new Error(errorMessage);
        }
        
        // Retry on 5xx errors (server errors)
        if (attempt < retries) {
          await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1))); // Exponential backoff
          continue;
        }
        
        throw new Error(errorMessage);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      
      // Handle network errors
      if (error.name === 'AbortError' || error.message.includes('Failed to fetch') || error.message.includes('NetworkError') || error.name === 'TypeError') {
        if (attempt < retries) {
          console.warn(`API request failed (attempt ${attempt + 1}/${retries + 1}), retrying...`, error.message);
          await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
          continue;
        }
        
        // Don't throw error for network failures - let the calling function handle fallback
        // Instead, return a special error object that can be checked
        const networkError = new Error(
          `Unable to connect to the server at ${API_BASE_URL}. Using offline mode.`
        );
        networkError.name = 'NetworkError';
        networkError.isNetworkError = true;
        throw networkError;
      }
      
      // Re-throw other errors
      if (attempt === retries) {
        console.error(`API Error (${endpoint}) after ${retries + 1} attempts:`, error);
        throw error;
      }
      
      // Wait before retry
      await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
    }
  }
}

/**
 * Test API connection
 */
export async function testAPI() {
  return apiRequest('/test/');
}

/**
 * Get all assessment questions with fallback
 */
export async function getQuestions() {
  try {
    const data = await apiRequest('/questions/');
    
    // Validate response structure
    if (data && data.questions && Array.isArray(data.questions) && data.questions.length > 0) {
      return data;
    }
    
    // If response is invalid, use fallback
    console.warn('Invalid response structure, using fallback questions');
    return {
      count: FALLBACK_QUESTIONS.length,
      questions: FALLBACK_QUESTIONS,
      fallback: true
    };
  } catch (error) {
    // Silently handle network errors - fallback will be used
    if (error.isNetworkError) {
      console.info('Backend server not available, using offline mode with fallback questions');
    } else {
      console.warn('Failed to fetch questions from API, using fallback:', error.message);
    }
    
    // Always return fallback questions if API fails - never throw
    return {
      count: FALLBACK_QUESTIONS.length,
      questions: FALLBACK_QUESTIONS,
      fallback: true,
      offline: true,
      error: error.isNetworkError ? 'Backend server not available' : error.message
    };
  }
}

/**
 * Analyze initial assessment answers
 */
export async function analyzeInitialAssessment(answers, assessmentId = null) {
  try {
    return await apiRequest('/analyze-initial/', {
      method: 'POST',
      body: JSON.stringify({
        answers,
        assessment_id: assessmentId,
      }),
    });
  } catch (error) {
    // Provide fallback analysis if API fails
    console.error('Analysis API failed, providing basic fallback:', error);
    
    // Basic fallback analysis
    const answerValues = Object.values(answers).map(v => parseInt(v) || 0);
    const totalScore = answerValues.reduce((sum, val) => sum + val, 0);
    const avgScore = answerValues.length > 0 ? totalScore / answerValues.length : 0;
    
    let riskLevel = 'low';
    if (avgScore >= 2.5) riskLevel = 'high';
    else if (avgScore >= 1.5) riskLevel = 'moderate';
    
    return {
      assessment_id: assessmentId || Date.now().toString(),
      analysis: {
        total_score: totalScore,
        average_score: avgScore.toFixed(2),
        primary_concerns: ['General assessment'],
        symptom_summary: 'Assessment completed in offline mode'
      },
      follow_up_questions: [
        'Can you tell me more about how you\'ve been feeling lately?',
        'What activities or situations have been most challenging for you?'
      ],
      risk_level: riskLevel,
      status: 'success',
      fallback: true,
      error: error.message
    };
  }
}

/**
 * Generate clinical report
 */
export async function generateClinicalReport(assessmentId, initialAnswers, followUpResponses) {
  try {
    return await apiRequest('/generate-report/', {
      method: 'POST',
      body: JSON.stringify({
        assessment_id: assessmentId,
        initial_answers: initialAnswers,
        follow_up_responses: followUpResponses,
      }),
    });
  } catch (error) {
    // Fallback report when backend is unavailable
    console.warn('Generate report failed, using fallback report:', error.message);
    return {
      assessment_id: assessmentId || Date.now().toString(),
      report: {
        summary: 'Offline mode: report generated locally.',
        initial_answers_count: initialAnswers ? Object.keys(initialAnswers).length : 0,
        follow_up_responses_count: followUpResponses ? Object.keys(followUpResponses).length : 0,
        recommendations: [
          'Please rerun when backend is online to get a full clinical report.',
          'Share these responses with a clinician for further review.',
        ],
      },
      status: 'success',
      assessment_complete: true,
      fallback: true,
      error: error.message,
    };
  }
}

/**
 * Get system status
 */
export async function getSystemStatus() {
  return apiRequest('/system-status/');
}

/**
 * Log in a user (Patient or Doctor)
 */
export async function login(username, password) {
  const url = `${API_URL}/auth/token/`;
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ username, password })
  });
  
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Invalid username or password');
  }
  
  const tokens = await response.json();
  localStorage.setItem('access_token', tokens.access);
  localStorage.setItem('refresh_token', tokens.refresh);
  
  // Fetch profile to verify role and cache profile info
  const profile = await getProfile();
  localStorage.setItem('user_role', profile.role);
  localStorage.setItem('user_name', `${profile.first_name} ${profile.last_name}`.trim() || profile.username);
  return profile;
}

/**
 * Register a new user
 */
export async function register(userData) {
  const url = `${API_URL}/auth/register/`;
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(userData)
  });
  
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || err.detail || 'Registration failed');
  }
  
  return await response.json();
}

/**
 * Get current user profile
 */
export async function getProfile() {
  const url = `${API_URL}/auth/me/`;
  const token = localStorage.getItem('access_token');
  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    }
  });
  
  if (!response.ok) {
    throw new Error('Failed to retrieve user profile');
  }
  
  return await response.json();
}

/**
 * Get Doctor Dashboard assessments and reports
 */
export async function getDoctorDashboard() {
  return await apiRequest('/doctor/dashboard/');
}

/**
 * Get Patient Dashboard assessments and history
 */
export async function getPatientDashboard() {
  return await apiRequest('/patient/dashboard/');
}

/**
 * Log out current user
 */
export function logout() {
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
  localStorage.removeItem('user_role');
  localStorage.removeItem('user_name');
}

/**
 * Delete a patient assessment record (Doctor only)
 */
export async function deleteAssessment(assessmentId) {
  return await apiRequest(`/doctor/assessments/${assessmentId}/delete/`, {
    method: 'DELETE',
  });
}

export { API_BASE_URL, API_URL, FALLBACK_QUESTIONS, checkBackendHealth };

