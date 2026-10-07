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
    location.replace("/static/login.html");
  },

  // 토큰이 있고 만료되지 않았는지 확인
  // JWT 가운데 부분(payload)은 누구나 읽을 수 있어서 exp(만료 시각)를 화면에서 바로 확인 가능
  // (위조 여부는 서버가 검사하므로 여기서는 "만료됐는지"만 본다)
  isLoggedIn() {
    const token = Auth.token;
    if (!token) return false;
    try {
      const payload = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
      const { exp } = JSON.parse(atob(payload));
      return exp * 1000 > Date.now();
    } catch {
      return false;
    }
  },

  // 로그인이 안 돼 있으면(또는 만료면) 로그인 화면으로. adminOnly면 관리자만 통과
  // <head>에서 호출하면 화면이 그려지기 전에 이동해서 다른 화면이 잠깐 보이지 않음
  // location.replace: 뒤로 가기를 눌러도 보호된 화면으로 돌아오지 않게
  require(adminOnly = false) {
    if (!Auth.isLoggedIn()) {
      Auth.clear();
      location.replace("/static/login.html");
      return false;
    }
    if (adminOnly && Auth.role !== "admin") { location.replace("/"); return false; }
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
