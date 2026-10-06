// 공통: 로그인 토큰 보관, API 호출, 로그인 확인
// 토큰은 브라우저 localStorage에 저장 (로그인 유지용)

const Auth = {
  get token() { return localStorage.getItem("token"); },
  get role() { return localStorage.getItem("role"); },
  get username() { return localStorage.getItem("username"); },

  save(token, role, username) {
    localStorage.setItem("token", token);
    localStorage.setItem("role", role);
    localStorage.setItem("username", username);
  },

  clear() {
    localStorage.removeItem("token");
    localStorage.removeItem("role");
    localStorage.removeItem("username");
  },

  logout() {
    Auth.clear();
    location.href = "/static/login.html";
  },

  // 로그인이 안 돼 있으면 로그인 화면으로. adminOnly면 관리자만 통과
  require(adminOnly = false) {
    if (!Auth.token) { location.href = "/static/login.html"; return false; }
    if (adminOnly && Auth.role !== "admin") { location.href = "/"; return false; }
    return true;
  },
};

// fetch를 감싼 함수: 토큰을 자동으로 붙이고, 401이면 로그인 화면으로 보냄
async function api(path, { method = "GET", body } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (Auth.token) headers["Authorization"] = "Bearer " + Auth.token;

  const res = await fetch(path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  if (res.status === 401 && path !== "/api/login") {
    Auth.logout();   // 토큰 만료 또는 위조 → 다시 로그인
    throw new Error("로그인이 필요합니다.");
  }

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    // FastAPI 오류 응답은 {"detail": "..."} 모양
    const detail = typeof data.detail === "string" ? data.detail : res.statusText;
    throw new Error(detail);
  }
  return data;
}
