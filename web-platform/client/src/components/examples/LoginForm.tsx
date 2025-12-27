import { LoginForm } from '../LoginForm';
import { ThemeProvider } from '../ThemeProvider';

export default function LoginFormExample() {
  return (
    <ThemeProvider>
      <div className="bg-background">
        <LoginForm 
          onLogin={(email, password) => console.log('Login:', { email, password })}
          onOIDCLogin={(provider) => console.log('OIDC Login:', provider)}
          loading={false}
        />
      </div>
    </ThemeProvider>
  );
}