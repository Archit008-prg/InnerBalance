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
    text: "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
    question_type: "scale",
    type: "scale",
    category: "depression",
    order: 1
  },
  {
    id: 2,
    text: "Over the last 2 weeks, how often have you been bothered by little interest or pleasure in doing things?",
    question_type: "scale",
    type: "scale",
    category: "depression",
    order: 2
  },
  {
    id: 3,
    text: "Over the last 2 weeks, how often have you been bothered by trouble falling or staying asleep, or sleeping too much?",
    question_type: "scale",
    type: "scale",
    category: "sleep",
    order: 3
  },
  {
    id: 4,
    text: "Over the last 2 weeks, how often have you been bothered by feeling tired or having little energy?",
    question_type: "scale",
    type: "scale",
    category: "energy",
    order: 4
  },
  {
    id: 5,
    text: "Over the last 2 weeks, how often have you been bothered by poor appetite or overeating?",
    question_type: "scale",
    type: "scale",
    category: "appetite",
    order: 5
  },
  {
    id: 6,
    text: "Over the last 2 weeks, how often have you been bothered by feeling bad about yourself, or that you are a failure, or have let yourself or your family down?",
    question_type: "scale",
    type: "scale",
    category: "mood",
    order: 6
  },
  {
    id: 7,
    text: "Over the last 2 weeks, how often have you been bothered by trouble concentrating on things, such as reading the newspaper or watching television?",
    question_type: "scale",
    type: "scale",
    category: "general",
    order: 7
  },
  {
    id: 8,
    text: "Over the last 2 weeks, how often have you been bothered by moving or speaking so slowly that other people could have noticed, or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?",
    question_type: "scale",
    type: "scale",
    category: "general",
    order: 8
  },
  {
    id: 9,
    text: "Over the last 2 weeks, how often have you been bothered by thoughts that you would be better off dead, or of hurting yourself?",
    question_type: "scale",
    type: "scale",
    category: "general",
    order: 9
  },
  {
    id: 10,
    text: "Over the last 2 weeks, how often have you been bothered by feeling nervous, anxious, or on edge?",
    question_type: "scale",
    type: "scale",
    category: "anxiety",
    order: 10
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
  const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
  
  const config = {
    headers: {
      'Content-Type': 'application/json',
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
        'What activities or situations have been most challenging for you?',
        'How long have you been experiencing these feelings?',
        'What support systems do you have in place?',
        'Have you noticed any changes in your daily routine?'
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

export { API_BASE_URL, API_URL, FALLBACK_QUESTIONS, checkBackendHealth };

